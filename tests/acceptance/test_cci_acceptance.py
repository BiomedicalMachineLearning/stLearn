"""Acceptance test of st.tl.cci.run on a synthetic Visium HD sample.

Checks the results match a stored baseline, and records time and memory so
they can be followed over time. It takes minutes, so pytest deselects it by default:

    # Baselines from the code checked out, e.g. master:
    pytest -m acceptance --acceptance-baseline=write

    # This checkout against the baselines, with a JUnit report:
    pytest -m acceptance --junitxml=tests/acceptance-results/junit.xml

Each scenario runs st.tl.cci.run in a fresh process (run_cci.py), so its peak
memory is its own. Run on an otherwise idle machine, and compare runs from the
same machine: times and memory depend on it.

Written to --acceptance-results (default tests/acceptance-results/):
- <scenario>.h5ad: this run's results and measurements (uns["metrics"]).
- history.csv: one row per scenario per run, appended, to follow time and
  memory over time.
- benchmark.json: this run's times and memory in github-action-benchmark's
  "customSmallerIsBetter" format.
The JUnit report gets the same measurements as test suite properties.
"""

import datetime
from pathlib import Path

import anndata
import numpy as np
import pytest

from stlearn.tl.cci import lr_pairs
from tests.acceptance import harness
from tests.acceptance.cci_sample import make_sample
from tests.acceptance.run_cci import NEIGHBOURS, RESULTS

# Scenario name: arguments for run_cci.py.
SCENARIOS = {
    "lit": [],
    "cell_types": ["--use-label"],
    "wide": ["--distance-um", "40"],
}
# Phases of cci.run timed in history.csv: phase name: column.
PHASES = {
    "neighbours": "phase_neighbours_seconds",
    "cell type counts": "phase_cell_type_counts_seconds",
    "LR scores": "phase_lr_scores_seconds",
    "testing": "phase_testing_seconds",
    "testing: backgrounds": "phase_testing_backgrounds_seconds",
}
CRASH_HINT = (
    "A crash in compiled code. For a readable traceback, rerun with "
    "NUMBA_NUM_THREADS=1 NUMBA_BOUNDSCHECK=1"
)


@pytest.fixture(scope="session")
def sample(request: pytest.FixtureRequest, tmp_path_factory) -> Path:
    """The synthetic sample, made once and kept in pytest's cache."""
    bins = request.config.getoption("--acceptance-bins")
    folder = harness.sample_folder(request, tmp_path_factory, "cci_acceptance")
    path = folder / f"sample_{bins}_bins.h5ad"
    if not path.exists():
        lr_genes = lr_pairs("connectomeDB2025").genes().tolist()
        make_sample(lr_genes, n_bins=bins).write_h5ad(path)
    return path


@pytest.fixture(scope="session")
def report(request: pytest.FixtureRequest):
    """Collects each scenario's row; writes benchmark.json and a summary at the end."""
    rows: list[dict] = []
    yield rows
    if not rows:
        return
    folder = Path(request.config.getoption("--acceptance-results"))
    harness.write_benchmark(folder / "benchmark.json", "cci.run", rows)
    lines = [
        f"{'scenario':12s}{'outcome':>18s}{'time (s)':>11s}{'vs base':>9s}"
        f"{'vs last':>9s}{'memory (GB)':>13s}{'vs base':>9s}{'vs last':>9s}"
    ]
    for row in rows:
        lines.append(
            f"{row['scenario']:12s}{row['outcome']:>18s}"
            f"{row['wall_seconds']:11.1f}{harness.times(row['time_vs_baseline']):>9s}"
            f"{harness.times(row['time_vs_last']):>9s}{row['peak_memory_gb']:13.2f}"
            f"{harness.times(row['memory_vs_baseline']):>9s}"
            f"{harness.times(row['memory_vs_last']):>9s}"
        )
    harness.write_summary(request, "CCI acceptance", lines, folder / "history.csv")


def run_scenario(sample, result, arguments, max_minutes, max_memory_gb):
    """Runs run_cci.py in a fresh process, stopping it if it goes over a limit."""
    arguments = [str(sample), str(result), *arguments]
    run = harness.run_script("run_cci", arguments, max_minutes, max_memory_gb)
    if run.failed:
        pytest.fail(harness.failure_message(run, CRASH_HINT), pytrace=False)
    return anndata.read_h5ad(result)


def differences(baseline: anndata.AnnData, current: anndata.AnnData, atol: float):
    """How the current results differ from the baseline, matching LRs by name."""
    if harness.metrics(baseline)["settings"] != harness.metrics(current)["settings"]:
        return ["the baseline used different settings: write a new one"]
    if not baseline.obs_names.equals(current.obs_names):
        return ["the samples' bins differ: write a new baseline"]
    lrs = baseline.uns["lr_summary"].index
    current_lrs = current.uns["lr_summary"].index
    if only := set(lrs) ^ set(current_lrs):
        return [f"different LR pairs tested ({len(only)} in only one run)"]
    order = current_lrs.get_indexer(lrs)

    problems = []
    for key in RESULTS:
        difference = abs(baseline.obsm[key] - current.obsm[key][:, order])
        if n := (difference > atol).nnz:
            largest = difference.max()
            problems.append(f"{key}: {n} values differ, by up to {largest:.3g}")

    summary = current.uns["lr_summary"].loc[lrs]
    if not np.array_equal(baseline.uns["lr_summary"].to_numpy(), summary.to_numpy()):
        problems.append("lr_summary differs")
    features = baseline.uns["lrfeatures"]
    current_features = current.uns["lrfeatures"].loc[features.index, features.columns]
    if not np.allclose(features, current_features, rtol=0, atol=atol, equal_nan=True):
        problems.append("lrfeatures differ")
    for key in NEIGHBOURS:
        before, now = baseline.uns.get(key), current.uns.get(key)
        if (before is None) != (now is None):
            problems.append(f"{key}: in only one run")
        elif before is not None and not np.array_equal(before, now):
            problems.append(f"{key}: the neighbourhoods differ")
    for key in ["cci_het"]:
        in_baseline, in_current = key in baseline.obs, key in current.obs
        if in_baseline != in_current:
            problems.append(f"{key}: in only one run")
        elif in_baseline:
            before = baseline.obs[key].astype(str).to_numpy()
            if n := (before != current.obs[key].astype(str).to_numpy()).sum():
                problems.append(f"{key}: {n} spots differ")
    return problems


def history_row(scenario: str, outcome: str, metrics: dict, baseline: dict | None):
    """A row of history.csv. It has every column, even without a baseline, as
    a new history.csv takes its header from the first row."""
    machine = metrics["machine"]
    seconds = metrics["wall_seconds"]
    memory_gb = metrics["peak_memory_bytes"] / harness.GB
    baseline_seconds = baseline_memory_gb = None
    if baseline is not None:
        baseline_seconds = baseline["wall_seconds"]
        baseline_memory_gb = baseline["peak_memory_bytes"] / harness.GB
    return {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "scenario": scenario,
        "outcome": outcome,
        "wall_seconds": seconds,
        "cpu_seconds": metrics["cpu_seconds"],
        "cores_busy": metrics["cpu_seconds"] / seconds,
        "peak_memory_gb": memory_gb,
        "memory_added_gb": metrics["memory_added_by_run_bytes"] / harness.GB,
        "warm_up_seconds": metrics["warm_up_seconds"],
        "baseline_wall_seconds": baseline_seconds,
        "baseline_peak_memory_gb": baseline_memory_gb,
        "time_vs_baseline": harness.ratio(seconds, baseline_seconds),
        "memory_vs_baseline": harness.ratio(memory_gb, baseline_memory_gb),
        **{
            column: metrics["phases"].get(phase, {}).get("wall_seconds")
            for phase, column in PHASES.items()
        },
        "bins": metrics["bins"],
        "genes": metrics["genes"],
        "lr_pairs_tested": metrics["lr_pairs_tested"],
        "n_pairs": metrics["settings"]["n_pairs"],
        "platform": machine["platform"],
        "processor": machine["processor"],
        "cpus": machine["cpus"],
        "numba_threads": machine["numba_threads"],
        "python": machine["python"],
        "numpy": machine["versions"]["numpy"],
        "numba": machine["versions"]["numba"],
    }


@pytest.mark.acceptance
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_cci_run(scenario, sample, report, request, record_testsuite_property):
    """st.tl.cci.run gives the baseline's results; its time and memory are kept."""
    option = request.config.getoption
    folder = Path(option("--acceptance-results"))
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / f"{scenario}.h5ad"

    current = run_scenario(
        sample,
        result_path,
        SCENARIOS[scenario],
        option("--acceptance-max-minutes"),
        option("--acceptance-max-memory-gb"),
    )
    outcome, problems, baseline_metrics = harness.against_baseline(
        current,
        result_path,
        harness.BASELINES / f"{scenario}.h5ad",
        option,
        differences,
    )

    row = history_row(scenario, outcome, harness.metrics(current), baseline_metrics)
    harness.record(
        row,
        folder / "history.csv",
        report,
        record_testsuite_property,
        scenario,
        ["wall_seconds", "cpu_seconds", "peak_memory_gb", "lr_pairs_tested"],
    )
    problems += harness.over_memory(
        row["peak_memory_gb"], option("--acceptance-max-memory-gb")
    )
    assert not problems, "\n".join(problems)
