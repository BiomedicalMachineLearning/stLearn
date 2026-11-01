"""Acceptance test of the spatial trajectory (pseudotime, then
pseudotimespace_global) on synthetic Visium HD samples.

Checks the results match a stored baseline, and records time and memory so
they can be followed over time. Each scenario has its own time and memory
limit (the --acceptance-max-* options can lower them), so a run that grows
out of hand, as in issue #340, is stopped and fails rather than running for
days. Deselected by default, like the CCI acceptance test:

    # Baselines from the code checked out, e.g. master:
    pytest -m acceptance -k trajectory --acceptance-baseline=write

    # This checkout against the baselines:
    pytest -m acceptance -k trajectory

Each scenario runs in a fresh process (run_trajectory.py), so its peak memory
is its own. Compare runs from the same machine: times and memory depend on it.

Written to --acceptance-results (default tests/acceptance-results/):
- trajectory_<scenario>.h5ad: this run's results and measurements.
- trajectory_history.csv: one row per scenario per run, appended, including
  runs stopped at a limit.
- trajectory_benchmark.json: this run's times and memory in
  github-action-benchmark's "customSmallerIsBetter" format.
"""

import csv
import datetime
import json
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import anndata
import numpy as np
import psutil
import pytest

from tests.acceptance.trajectory_sample import BIN_SPACING, make_sample

BASELINES = Path(__file__).parent / "baselines"
ROOT = Path(__file__).resolve().parents[2]
GB = 1024**3
EPS = 3 * BIN_SPACING  # subclusters: bins within three bins of each other


@dataclass(frozen=True)
class Scenario:
    n_clusters: int
    arguments: tuple[str, ...]
    max_minutes: float
    max_memory_gb: float


SCENARIOS = {
    # The whole workflow, including pseudotimespace_global on every stored
    # path as the report in #340 did. Takes seconds.
    "clusters_12": Scenario(12, ("--all-lineages",), max_minutes=5, max_memory_gb=4),
    # Many clusters that PAGA connects to many others, as in #340: storing
    # the paths of up to 3 clusters (3,088 here) currently walks every path
    # between every two clusters, about a billion, which takes hours.
    "clusters_30": Scenario(30, (), max_minutes=3, max_memory_gb=4),
}
KNOWN_ISSUES = {
    "clusters_30": "#340: store_available_paths walks every path, of any length",
}
# Phases timed in the history: phase name: column.
PHASES = {
    "subclusters": "phase_subclusters_seconds",
    "PAGA": "phase_paga_seconds",
    "diffusion map": "phase_diffmap_seconds",
    "DPT": "phase_dpt_seconds",
    "available paths": "phase_available_paths_seconds",
    "global level": "phase_global_level_seconds",
}


def _scenarios() -> list:
    """The scenarios, a known issue's marked to fail while it is open."""
    params = []
    for name in SCENARIOS:
        marks = []
        if name in KNOWN_ISSUES:
            # Strict: once the run finishes in time this fails, as a reminder
            # to drop the mark and write the scenario's baseline.
            marks.append(
                pytest.mark.xfail(
                    reason=KNOWN_ISSUES[name], raises=pytest.fail.Exception, strict=True
                )
            )
        params.append(pytest.param(name, marks=marks))
    return params


@pytest.fixture(scope="session")
def samples(request: pytest.FixtureRequest, tmp_path_factory):
    """Makes each scenario's sample once, kept in pytest's cache."""
    bins = request.config.getoption("--acceptance-bins")
    if getattr(request.config, "cache", None) is not None:
        folder = request.config.cache.mkdir("trajectory_acceptance")
    else:
        folder = tmp_path_factory.mktemp("trajectory_acceptance")

    def sample(n_clusters: int) -> Path:
        path = folder / f"sample_{bins}_bins_{n_clusters}_clusters.h5ad"
        if not path.exists():
            make_sample(n_bins=bins, n_clusters=n_clusters).write_h5ad(path)
        return path

    return sample


@pytest.fixture(scope="session")
def report(request: pytest.FixtureRequest):
    """Collects each scenario's row; writes the benchmark and a summary at the end."""
    rows: list[dict] = []
    yield rows
    if not rows:
        return
    folder = Path(request.config.getoption("--acceptance-results"))
    benchmarks = [
        {
            "name": f"trajectory {row['scenario']}: {name}",
            "unit": unit,
            "value": row[key],
        }
        for row in rows
        for name, unit, key in [
            ("time", "s", "wall_seconds"),
            ("peak memory", "GB", "peak_memory_gb"),
        ]
    ]
    (folder / "trajectory_benchmark.json").write_text(json.dumps(benchmarks, indent=2))

    plugins = request.config.pluginmanager
    terminal = plugins.get_plugin("terminalreporter")
    capture = plugins.get_plugin("capturemanager")
    if terminal is None or capture is None:
        return
    with capture.global_and_fixture_disabled():
        terminal.write_line("")
        terminal.write_sep("=", "Trajectory acceptance")
        terminal.write_line(
            f"{'scenario':13s}{'outcome':>22s}{'time (s)':>10s}{'vs base':>9s}"
            f"{'vs last':>9s}{'memory (GB)':>13s}{'paths':>8s}"
        )
        for row in rows:
            paths = (
                "-" if row["available_paths"] is None else str(row["available_paths"])
            )
            terminal.write_line(
                f"{row['scenario']:13s}{row['outcome']:>22s}{row['wall_seconds']:10.1f}"
                f"{_times(row['time_vs_baseline']):>9s}{_times(row['time_vs_last']):>9s}"
                f"{row['peak_memory_gb']:13.2f}{paths:>8s}"
            )
        terminal.write_line(f"History: {folder / 'trajectory_history.csv'}")


def _metrics(run: anndata.AnnData) -> dict:
    """run_trajectory.py's measurements and settings for the run."""
    return json.loads(run.uns["metrics"])


def _times(ratio: float | None) -> str:
    return "-" if ratio is None else f"{ratio:.2f}x"


def _ratio(value: float, earlier: float | str | None) -> float | None:
    """value / earlier, or None without an earlier value (CSV text is fine)."""
    return None if earlier in (None, "") else value / float(earlier)


@dataclass
class Run:
    """How run_trajectory.py ended: its command and output, how long it ran,
    the most memory seen, and why it was stopped, if it was."""

    command: list[str]
    finished: subprocess.CompletedProcess
    seconds: float
    memory_gb: float
    stopped: str | None


def run_scenario(sample, result, arguments, max_minutes, max_memory_gb) -> Run:
    """Runs run_trajectory.py in a fresh process, stopping it if it goes over
    a limit."""
    command = [sys.executable, "-m", "tests.acceptance.run_trajectory", str(sample)]
    command += [str(result), "--eps", str(EPS), *arguments]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    child, started = psutil.Process(process.pid), time.monotonic()
    stopped, memory_gb = None, 0.0
    while True:
        try:  # waits up to a second; retrying keeps the output read so far
            stdout, stderr = process.communicate(timeout=1)
            break
        except subprocess.TimeoutExpired:
            pass
        minutes = (time.monotonic() - started) / 60
        try:
            memory_gb = max(memory_gb, child.memory_info().rss / GB)
        except psutil.NoSuchProcess:  # it has just finished
            continue
        if minutes > max_minutes:
            stopped = (
                f"stopped after {minutes:.1f} minutes, "
                f"over the {max_minutes:g} minute limit"
            )
        elif memory_gb > max_memory_gb:
            stopped = (
                f"stopped at {memory_gb:.1f} GB, over the {max_memory_gb:g} GB limit"
            )
        if stopped:
            process.kill()
            stdout, stderr = process.communicate()
            break
    seconds = time.monotonic() - started
    finished = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    return Run(command, finished, seconds, memory_gb, stopped)


def _failure_message(run: Run) -> str:
    """How run_trajectory.py ended, how to rerun it, and what it printed."""
    code = run.finished.returncode
    if run.stopped:
        lines = [f"run_trajectory.py was {run.stopped}"]
    elif code >= 0:
        lines = [f"run_trajectory.py exited with code {code}"]
    else:
        try:
            how = signal.Signals(-code).name
        except ValueError:
            how = f"signal {-code}"
        lines = [f"run_trajectory.py was killed by {how}"]
    lines.append(f"Rerun: cd {shlex.quote(str(ROOT))} && {shlex.join(run.command)}")
    for stream, text in [
        ("stdout", run.finished.stdout),
        ("stderr", run.finished.stderr),
    ]:
        text = text.strip()
        lines.append(f"--- {stream} ---\n{text[-5000:] if text else '(empty)'}")
    return "\n".join(lines)


def differences(baseline: anndata.AnnData, current: anndata.AnnData, atol: float):
    """How the current results differ from the baseline."""
    if _metrics(baseline)["settings"] != _metrics(current)["settings"]:
        return ["the baseline used different settings: write a new one"]
    if not baseline.obs_names.equals(current.obs_names):
        return ["the samples' bins differ: write a new baseline"]

    problems = []
    before = baseline.obs["sub_cluster_labels"].astype(str).to_numpy()
    if n := (before != current.obs["sub_cluster_labels"].astype(str).to_numpy()).sum():
        problems.append(f"sub_cluster_labels: {n} bins differ")
    pseudotime = [run.obs["dpt_pseudotime"].to_numpy() for run in (baseline, current)]
    if not np.allclose(*pseudotime, rtol=0, atol=atol, equal_nan=True):
        largest = np.nanmax(np.abs(pseudotime[0] - pseudotime[1]))
        problems.append(f"dpt_pseudotime differs, by up to {largest:.3g}")
    if baseline.uns["split_node"] != current.uns["split_node"]:
        problems.append("split_node differs")

    # Paths are compared as a set, not by key: see run_trajectory.py.
    paths = [
        set(map(tuple, json.loads(run.uns["available_paths"])))
        for run in (baseline, current)
    ]
    if paths[0] != paths[1]:
        problems.append(
            f"available_paths: {len(paths[0] - paths[1])} only in the baseline, "
            f"{len(paths[1] - paths[0])} only in this run"
        )

    edges = [json.loads(run.uns["pts_graph"]) for run in (baseline, current)]
    if [e[:2] for e in edges[0]] != [e[:2] for e in edges[1]]:
        problems.append("PTS_graph has different edges")
    elif not np.allclose(
        [e[2] for e in edges[0]], [e[2] for e in edges[1]], rtol=0, atol=atol
    ):
        problems.append("PTS_graph edge weights differ")

    outcomes = [_metrics(run)["lineage_outcomes"] for run in (baseline, current)]
    if outcomes[0] != outcomes[1]:
        problems.append(
            f"pseudotimespace_global on every path ended differently: "
            f"baseline {outcomes[0]}, this run {outcomes[1]}"
        )
    return problems


def history_row(scenario: str, outcome: str, run: Run, metrics: dict | None, baseline):
    """A row of trajectory_history.csv. It has every column, even for a run
    stopped at a limit, as a new file takes its header from the first row."""
    metrics = metrics or {}
    machine = metrics.get("machine", {})
    seconds = metrics.get("wall_seconds", run.seconds)
    memory_gb = metrics["peak_memory_bytes"] / GB if metrics else run.memory_gb
    baseline_seconds = baseline_memory_gb = None
    if baseline is not None:
        baseline_seconds = baseline["wall_seconds"]
        baseline_memory_gb = baseline["peak_memory_bytes"] / GB
    phases = metrics.get("phases", {})
    return {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "scenario": scenario,
        "outcome": outcome,
        "wall_seconds": seconds,
        "cpu_seconds": metrics.get("cpu_seconds"),
        "peak_memory_gb": memory_gb,
        "pseudotime_seconds": metrics.get("pseudotime_seconds"),
        "pseudotimespace_global_seconds": metrics.get("pseudotimespace_global_seconds"),
        "warm_up_seconds": metrics.get("warm_up_seconds"),
        "baseline_wall_seconds": baseline_seconds,
        "baseline_peak_memory_gb": baseline_memory_gb,
        "time_vs_baseline": _ratio(seconds, baseline_seconds),
        "memory_vs_baseline": _ratio(memory_gb, baseline_memory_gb),
        **{
            column: phases.get(phase, {}).get("wall_seconds")
            for phase, column in PHASES.items()
        },
        "bins": metrics.get("bins"),
        "clusters": SCENARIOS[scenario].n_clusters,
        "subclusters_kept": metrics.get("subclusters_kept"),
        "available_paths": metrics.get("available_paths"),
        "lineage_outcomes": json.dumps(metrics.get("lineage_outcomes", {})),
        "platform": machine.get("platform", sys.platform),
        "cpus": machine.get("cpus"),
        "python": machine.get("python"),
        "networkx": machine.get("versions", {}).get("networkx"),
        "scanpy": machine.get("versions", {}).get("scanpy"),
    }


def append_history(path: Path, row: dict) -> dict:
    """Appends the row to the history; the scenario's previous run on this
    machine, or {} if there is none."""
    earlier, columns = [], None
    if path.exists():
        with open(path, newline="") as file:
            reader = csv.DictReader(file)
            earlier, columns = list(reader), reader.fieldnames
    with open(path, "a", newline="") as file:
        # An existing file keeps its header, so the row lines up with it.
        writer = csv.DictWriter(file, columns or list(row), extrasaction="ignore")
        if columns is None:
            writer.writeheader()
        writer.writerow(row)
    same = [
        run
        for run in earlier
        if run["scenario"] == row["scenario"] and run["platform"] == row["platform"]
    ]
    return same[-1] if same else {}


@pytest.mark.acceptance
@pytest.mark.parametrize("scenario", _scenarios())
def test_trajectory(scenario, samples, report, request, record_testsuite_property):
    """The trajectory gives the baseline's results, within the scenario's time
    and memory limits; its time and memory are kept."""
    option = request.config.getoption
    settings = SCENARIOS[scenario]
    folder = Path(option("--acceptance-results"))
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / f"trajectory_{scenario}.h5ad"
    baseline_path = BASELINES / f"trajectory_{scenario}.h5ad"
    max_minutes = min(
        settings.max_minutes, option("--acceptance-max-minutes") or np.inf
    )
    max_memory_gb = min(
        settings.max_memory_gb, option("--acceptance-max-memory-gb") or np.inf
    )

    run = run_scenario(
        samples(settings.n_clusters),
        result_path,
        settings.arguments,
        max_minutes,
        max_memory_gb,
    )
    current = metrics = None
    problems, baseline_metrics = [], None
    if run.stopped or run.finished.returncode != 0:
        outcome = "stopped at limit" if run.stopped else "failed"
    else:
        current = anndata.read_h5ad(result_path)
        metrics = _metrics(current)
        if option("--acceptance-baseline") == "write":
            BASELINES.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(result_path, baseline_path)
            outcome = "baseline written"
        elif not baseline_path.exists():
            outcome = "no baseline"
            problems.append(
                f"no baseline {baseline_path}: "
                "write one with --acceptance-baseline=write",
            )
        else:
            baseline = anndata.read_h5ad(baseline_path)
            baseline_metrics = _metrics(baseline)
            problems = differences(baseline, current, option("--acceptance-atol"))
            outcome = "results differ" if problems else "results identical"

    row = history_row(scenario, outcome, run, metrics, baseline_metrics)
    last = append_history(folder / "trajectory_history.csv", row)
    row["time_vs_last"] = _ratio(row["wall_seconds"], last.get("wall_seconds"))
    row["memory_vs_last"] = _ratio(row["peak_memory_gb"], last.get("peak_memory_gb"))
    report.append(row)
    for key in ["wall_seconds", "peak_memory_gb", "available_paths"]:
        record_testsuite_property(f"trajectory_{scenario}.{key}", row[key])

    if current is None:
        pytest.fail(_failure_message(run), pytrace=False)
    if row["peak_memory_gb"] > max_memory_gb:
        problems.append(
            f"peak memory {row['peak_memory_gb']:.1f} GB, "
            f"over the {max_memory_gb:g} GB limit",
        )
    assert not problems, "\n".join(problems)
