"""Acceptance test of st.tl.cci.run on a synthetic Visium HD sample.

Checks the results match a stored baseline, and records time and memory so
they can be followed over time. It takes minutes, so `pytest` skips it:

    # Baselines from master's code, using a git worktree of master:
    pytest -m acceptance --acceptance-baseline=write --acceptance-stlearn .

    # This checkout against the baselines, with a JUnit report:
    pytest -m acceptance --junitxml=acceptance-results/junit.xml

Each scenario runs st.tl.cci.run in a fresh process (run_cci.py), so its peak
memory is its own. Run on an otherwise idle machine, and compare runs from the
same machine: times and memory depend on it.

Written to --acceptance-results (default acceptance-results/):
- <scenario>.h5ad: this run's results and measurements (uns["metrics"]).
- history.csv: one row per scenario per run, appended, to follow time and
  memory over time.
- benchmark.json: this run's times and memory in github-action-benchmark's
  "customSmallerIsBetter" format.
The JUnit report gets the same measurements as test suite properties.

Baselines are tests/acceptance/baselines/<scenario>.h5ad, the same format.
"""

import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import anndata
import numpy as np
import pytest

from stlearn.tl.cci.analysis import load_lrs
from tests.acceptance.cci_sample import make_sample
from tests.acceptance.run_cci import NEIGHBOURS, RESULTS

BASELINES = Path(__file__).parent / "baselines"
ROOT = Path(__file__).resolve().parents[2]
GB = 1024**3
# Scenario name: arguments for run_cci.py.
SCENARIOS = {
    "lit": [],
    "cell_types": ["--use-label"],
}
HISTORY_COLUMNS = [
    "timestamp",
    "scenario",
    "outcome",
    "commit",
    "branch",
    "uncommitted_changes",
    "wall_seconds",
    "cpu_seconds",
    "cores_busy",
    "peak_memory_gb",
    "memory_added_gb",
    "warm_up_seconds",
    "baseline_wall_seconds",
    "baseline_peak_memory_gb",
    "time_vs_baseline",
    "memory_vs_baseline",
    "phase_neighbours_seconds",
    "phase_cell_type_counts_seconds",
    "phase_lr_scores_seconds",
    "phase_testing_seconds",
    "phase_testing_backgrounds_seconds",
    "bins",
    "genes",
    "lr_pairs_tested",
    "n_pairs",
    "platform",
    "processor",
    "cpus",
    "numba_threads",
    "python",
    "numpy",
    "numba",
    "stlearn",
]


@pytest.fixture(scope="session")
def sample(request: pytest.FixtureRequest, tmp_path_factory) -> Path:
    """The synthetic sample, made once and kept in pytest's cache."""
    bins = request.config.getoption("--acceptance-bins")
    if getattr(request.config, "cache", None) is not None:
        folder = request.config.cache.mkdir("cci_acceptance")
    else:
        folder = tmp_path_factory.mktemp("cci_acceptance")
    path = folder / f"sample_{bins}_bins.h5ad"
    if not path.exists():
        lrs = load_lrs(["connectomeDB2020_lit"], species="human")
        lr_genes = sorted({gene for lr in lrs for gene in lr.split("_")})
        make_sample(lr_genes, n_bins=bins).write_h5ad(path)
    return path


@pytest.fixture(scope="session")
def report(request: pytest.FixtureRequest):
    """Collects each scenario's row; writes benchmark.json and a summary at the end."""
    rows = []
    yield rows
    if not rows:
        return
    folder = Path(request.config.getoption("--acceptance-results"))
    benchmarks = []
    for row in rows:
        name = f"cci.run {row['scenario']}"
        benchmarks.append(
            {"name": f"{name}: time", "unit": "s", "value": row["wall_seconds"]}
        )
        benchmarks.append(
            {
                "name": f"{name}: peak memory",
                "unit": "GB",
                "value": row["peak_memory_gb"],
            }
        )
    (folder / "benchmark.json").write_text(json.dumps(benchmarks, indent=2))

    plugins = request.config.pluginmanager
    terminal = plugins.get_plugin("terminalreporter")
    capture = plugins.get_plugin("capturemanager")
    if terminal is None or capture is None:
        return
    with capture.global_and_fixture_disabled():
        terminal.write_line("")
        terminal.write_sep("=", "CCI acceptance")
        terminal.write_line(
            f"{'scenario':12s}{'outcome':>18s}{'time (s)':>11s}{'vs base':>9s}"
            f"{'vs last':>9s}{'memory (GB)':>13s}{'vs base':>9s}{'vs last':>9s}"
        )
        for row in rows:
            terminal.write_line(
                f"{row['scenario']:12s}{row['outcome']:>18s}"
                f"{row['wall_seconds']:11.1f}{_ratio(row, 'time_vs_baseline'):>9s}"
                f"{_ratio(row, 'time_vs_last'):>9s}{row['peak_memory_gb']:13.2f}"
                f"{_ratio(row, 'memory_vs_baseline'):>9s}"
                f"{_ratio(row, 'memory_vs_last'):>9s}"
            )
        terminal.write_line(f"History: {folder / 'history.csv'}")


def _ratio(row: dict, key: str) -> str:
    value = row.get(key)
    return f"{value:.2f}x" if value not in (None, "") else "-"


def run_scenario(sample: Path, result: Path, arguments: list[str], stlearn) -> dict:
    """Runs run_cci.py in a fresh process; its measurements."""
    command = [sys.executable, "-m", "tests.acceptance.run_cci", str(sample)]
    command += [str(result), *arguments]
    if stlearn:
        command += ["--stlearn", str(Path(stlearn).resolve())]
    path = [str(ROOT), os.environ.get("PYTHONPATH", "")]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, path))}
    finished = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True
    )
    if finished.returncode != 0:
        pytest.fail(f"run_cci.py failed:\n{finished.stderr[-5000:]}")
    return json.loads(anndata.read_h5ad(result).uns["metrics"])


def differences(baseline: anndata.AnnData, current: anndata.AnnData, atol: float):
    """How the current results differ from the baseline, matching LRs by name."""
    lrs = baseline.uns["lr_summary"].index
    current_lrs = current.uns["lr_summary"].index
    if not baseline.obs_names.equals(current.obs_names):
        return ["the samples' bins differ: write a new baseline"]
    if set(lrs) != set(current_lrs):
        only = set(lrs).symmetric_difference(current_lrs)
        return [f"different LR pairs tested ({len(only)} in only one run)"]
    order = current_lrs.get_indexer(lrs)

    problems = []
    for key in RESULTS:
        difference = abs(baseline.obsm[key] - current.obsm[key][:, order])
        largest = difference.max() if difference.nnz else 0
        if largest > atol:
            n = (difference > atol).nnz
            problems.append(f"{key}: {n} values differ, by up to {largest:.3g}")

    summary = current.uns["lr_summary"].loc[lrs]
    if not np.array_equal(baseline.uns["lr_summary"].to_numpy(), summary.to_numpy()):
        problems.append("lr_summary differs")
    features = baseline.uns["lrfeatures"]
    current_features = current.uns["lrfeatures"].loc[features.index, features.columns]
    if not np.allclose(features, current_features, rtol=0, atol=atol, equal_nan=True):
        problems.append("lrfeatures differ")
    for key in [*NEIGHBOURS, "cci_het"]:
        if key in baseline.obs or key in current.obs:
            if key not in baseline.obs or key not in current.obs:
                problems.append(f"{key}: in only one run")
            else:
                before = baseline.obs[key].astype(str).to_numpy()
                n = (before != current.obs[key].astype(str).to_numpy()).sum()
                if n:
                    problems.append(f"{key}: {n} spots differ")
    return problems


def history_row(scenario: str, outcome: str, metrics: dict, baseline: dict | None):
    machine = metrics["machine"]
    phases = metrics["phases"]
    row = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "scenario": scenario,
        "outcome": outcome,
        "commit": metrics["commit"],
        "branch": metrics["branch"],
        "uncommitted_changes": metrics["uncommitted_changes"],
        "wall_seconds": metrics["wall_seconds"],
        "cpu_seconds": metrics["cpu_seconds"],
        "cores_busy": metrics["cpu_seconds"] / metrics["wall_seconds"],
        "peak_memory_gb": metrics["peak_memory_bytes"] / GB,
        "memory_added_gb": metrics["memory_added_by_run_bytes"] / GB,
        "warm_up_seconds": metrics["warm_up_seconds"],
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
        "stlearn": metrics["stlearn"],
    }
    for name, column in [
        ("neighbours", "phase_neighbours_seconds"),
        ("cell type counts", "phase_cell_type_counts_seconds"),
        ("LR scores", "phase_lr_scores_seconds"),
        ("testing", "phase_testing_seconds"),
        ("testing: backgrounds", "phase_testing_backgrounds_seconds"),
    ]:
        row[column] = phases.get(name, {}).get("wall_seconds", "")
    if baseline is not None:
        row["baseline_wall_seconds"] = baseline["wall_seconds"]
        row["baseline_peak_memory_gb"] = baseline["peak_memory_bytes"] / GB
        row["time_vs_baseline"] = metrics["wall_seconds"] / baseline["wall_seconds"]
        row["memory_vs_baseline"] = (
            metrics["peak_memory_bytes"] / baseline["peak_memory_bytes"]
        )
    return row


def record_history(path: Path, row: dict) -> dict | None:
    """Appends the row to the history; the previous run on this machine, if any."""
    previous = None
    if path.exists():
        with open(path, newline="") as file:
            for earlier in csv.DictReader(file):
                same_machine = earlier["platform"] == row["platform"]
                if earlier["scenario"] == row["scenario"] and same_machine:
                    previous = earlier
    new_file = not path.exists()
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return previous


@pytest.mark.acceptance
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_cci_run(scenario, sample, report, request, record_testsuite_property):
    """st.tl.cci.run gives the baseline's results; its time and memory are kept."""
    option = request.config.getoption
    folder = Path(option("--acceptance-results"))
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / f"{scenario}.h5ad"
    baseline_path = BASELINES / f"{scenario}.h5ad"

    metrics = run_scenario(
        sample, result_path, SCENARIOS[scenario], option("--acceptance-stlearn")
    )
    current = anndata.read_h5ad(result_path)

    problems, baseline_metrics = [], None
    if option("--acceptance-baseline") == "write":
        BASELINES.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(result_path, baseline_path)
        outcome = "baseline written"
    elif not baseline_path.exists():
        outcome = "no baseline"
        problems.append(
            f"no baseline {baseline_path}: write one, e.g. from master, with "
            "--acceptance-baseline=write --acceptance-stlearn PATH"
        )
    else:
        baseline = anndata.read_h5ad(baseline_path)
        baseline_metrics = json.loads(baseline.uns["metrics"])
        if baseline_metrics["settings"] != metrics["settings"]:
            problems.append("the baseline used different settings: write a new one")
        else:
            problems += differences(baseline, current, option("--acceptance-atol"))
        outcome = "results differ" if problems else "results identical"

    row = history_row(scenario, outcome, metrics, baseline_metrics)
    previous = record_history(folder / "history.csv", row)
    if previous is not None:
        row["time_vs_last"] = row["wall_seconds"] / float(previous["wall_seconds"])
        row["memory_vs_last"] = row["peak_memory_gb"] / float(
            previous["peak_memory_gb"]
        )
    report.append(row)
    for key in ["wall_seconds", "cpu_seconds", "peak_memory_gb", "lr_pairs_tested"]:
        record_testsuite_property(f"{scenario}.{key}", row[key])

    for ratio, limit in [
        ("time_vs_baseline", option("--acceptance-max-time-ratio")),
        ("memory_vs_baseline", option("--acceptance-max-memory-ratio")),
    ]:
        if limit is not None and ratio in row and row[ratio] > limit:
            problems.append(f"{ratio} is {row[ratio]:.2f}, over the limit {limit}")
    assert not problems, "\n".join(problems)
