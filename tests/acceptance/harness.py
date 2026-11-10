"""What the acceptance tests share: running a scenario's script in a fresh
process within time and memory limits, writing or comparing its baseline, and
keeping its time and memory in a history, a benchmark and a summary."""

import csv
import json
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import anndata
import psutil
import pytest

BASELINES = Path(__file__).parent / "baselines"
ROOT = Path(__file__).resolve().parents[2]
GB = 1024**3


def metrics(run: anndata.AnnData) -> dict:
    """The script's measurements and settings for the run."""
    return json.loads(run.uns["metrics"])


def times(ratio: float | None) -> str:
    return "-" if ratio is None else f"{ratio:.2f}x"


def ratio(value: float, earlier: float | str | None) -> float | None:
    """value / earlier, or None without an earlier value (CSV text is fine)."""
    return None if earlier in (None, "") else value / float(earlier)


def sample_folder(request: pytest.FixtureRequest, tmp_path_factory, name: str) -> Path:
    """Where the samples are kept: pytest's cache, or a temporary folder without
    it."""
    if getattr(request.config, "cache", None) is not None:
        return request.config.cache.mkdir(name)
    return tmp_path_factory.mktemp(name)


@dataclass
class Run:
    """How a script ended: its command and output, how long it ran, the most
    memory seen, and why it was stopped, if it was."""

    script: str
    command: list[str]
    finished: subprocess.CompletedProcess
    seconds: float
    memory_gb: float
    stopped: str | None

    @property
    def failed(self) -> bool:
        return bool(self.stopped) or self.finished.returncode != 0


def run_script(
    script: str,
    arguments: list[str],
    max_minutes: float | None,
    max_memory_gb: float | None,
) -> Run:
    """Runs tests/acceptance/<script>.py in a fresh process, stopping it if it
    goes over a limit (None for no limit)."""
    command = [sys.executable, "-m", f"tests.acceptance.{script}", *arguments]
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
        if max_minutes is not None and minutes > max_minutes:
            stopped = (
                f"stopped after {minutes:.1f} minutes, "
                f"over the {max_minutes:g} minute limit"
            )
        elif max_memory_gb is not None and memory_gb > max_memory_gb:
            stopped = (
                f"stopped at {memory_gb:.1f} GB, over the {max_memory_gb:g} GB limit"
            )
        if stopped:
            process.kill()
            stdout, stderr = process.communicate()
            break
    seconds = time.monotonic() - started
    finished = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    return Run(f"{script}.py", command, finished, seconds, memory_gb, stopped)


def failure_message(run: Run, crash_hint: str | None = None) -> str:
    """How the script ended, how to rerun it, and what it printed; `crash_hint`
    too if it crashed (SIGSEGV, SIGBUS or SIGABRT)."""
    code = run.finished.returncode
    if run.stopped:
        lines = [f"{run.script} was {run.stopped}"]
    elif code >= 0:
        lines = [f"{run.script} exited with code {code}"]
    else:
        try:
            how = signal.Signals(-code).name
        except ValueError:
            how = f"signal {-code}"
        lines = [f"{run.script} was killed by {how}"]
        if crash_hint and how in ("SIGSEGV", "SIGBUS", "SIGABRT"):
            lines.append(crash_hint)
    lines.append(f"Rerun: cd {shlex.quote(str(ROOT))} && {shlex.join(run.command)}")
    for stream, text in [
        ("stdout", run.finished.stdout),
        ("stderr", run.finished.stderr),
    ]:
        text = text.strip()
        lines.append(f"--- {stream} ---\n{text[-5000:] if text else '(empty)'}")
    return "\n".join(lines)


def against_baseline(
    current: anndata.AnnData,
    result_path: Path,
    baseline_path: Path,
    option: Callable,
    differences: Callable[[anndata.AnnData, anndata.AnnData, float], list[str]],
) -> tuple[str, list[str], dict | None]:
    """Writes the result as the baseline, with --acceptance-baseline=write, or
    compares it with the baseline. The outcome, the problems found, and the
    baseline's metrics when there was one to compare with."""
    if option("--acceptance-baseline") == "write":
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(result_path, baseline_path)
        return "baseline written", [], None
    if not baseline_path.exists():
        problem = (
            f"no baseline {baseline_path}: write one with --acceptance-baseline=write"
        )
        return "no baseline", [problem], None
    baseline = anndata.read_h5ad(baseline_path)
    problems = differences(baseline, current, option("--acceptance-atol"))
    outcome = "results differ" if problems else "results identical"
    return outcome, problems, metrics(baseline)


def over_memory(peak_memory_gb: float, limit: float | None) -> list[str]:
    """The problem, if the peak memory went over the limit."""
    if limit is None or peak_memory_gb <= limit:
        return []
    return [f"peak memory {peak_memory_gb:.1f} GB, over the {limit:g} GB limit"]


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


def record(
    row: dict,
    history: Path,
    report: list[dict],
    record_testsuite_property: Callable,
    prefix: str,
    properties: list[str],
) -> None:
    """Appends the scenario's row to the history, and to the session's report
    with its time and memory against its last run on this machine. Its
    `properties` go in the JUnit report too, as prefix.property."""
    last = append_history(history, row)
    row["time_vs_last"] = ratio(row["wall_seconds"], last.get("wall_seconds"))
    row["memory_vs_last"] = ratio(row["peak_memory_gb"], last.get("peak_memory_gb"))
    report.append(row)
    for key in properties:
        record_testsuite_property(f"{prefix}.{key}", row[key])


def write_benchmark(path: Path, label: str, rows: list[dict]) -> None:
    """Each scenario's time and peak memory in github-action-benchmark's
    "customSmallerIsBetter" format."""
    benchmarks = [
        {"name": f"{label} {row['scenario']}: {name}", "unit": unit, "value": row[key]}
        for row in rows
        for name, unit, key in [
            ("time", "s", "wall_seconds"),
            ("peak memory", "GB", "peak_memory_gb"),
        ]
    ]
    path.write_text(json.dumps(benchmarks, indent=2))


def write_summary(
    request: pytest.FixtureRequest, title: str, lines: list[str], history: Path
) -> None:
    """Prints the lines under the title, past pytest's capture, and where the
    history is."""
    plugins = request.config.pluginmanager
    terminal = plugins.get_plugin("terminalreporter")
    capture = plugins.get_plugin("capturemanager")
    if terminal is None or capture is None:
        return
    with capture.global_and_fixture_disabled():
        terminal.write_line("")
        terminal.write_sep("=", title)
        for line in lines:
            terminal.write_line(line)
        terminal.write_line(f"History: {history}")
