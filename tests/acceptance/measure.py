"""Measurements the acceptance tests' scripts (run_cci.py, run_trajectory.py) take
of their run: its peak memory, the time of each of its phases, and the machine
it ran on, saved with its results."""

import importlib
import json
import os
import pathlib
import platform
import resource
import sys
import time
from types import ModuleType
from typing import Any

import anndata


def peak_memory_bytes() -> int:
    """This process's peak resident memory so far."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # Linux: kilobytes


class PhaseTimer:
    """Wall and CPU time of each phase, by wrapping the functions it calls:
    `phases` gives each phase's (module, function).

    CPU time counts every thread, so CPU / wall is the average number of busy
    cores.
    """

    def __init__(self, phases: dict[str, tuple[str, str]]) -> None:
        self.phases = phases
        self.totals = {name: [0.0, 0.0, 0] for name in phases}
        self._originals: list[tuple[ModuleType, str, Any]] = []

    def __enter__(self) -> "PhaseTimer":
        for name, (module_name, attribute) in self.phases.items():
            module = importlib.import_module(module_name)
            original = getattr(module, attribute)
            setattr(module, attribute, self._timed(name, original))
            self._originals.append((module, attribute, original))
        return self

    def __exit__(self, *exc) -> None:
        for module, attribute, original in reversed(self._originals):
            setattr(module, attribute, original)

    def _timed(self, name, function):
        def wrapper(*args, **kwargs):
            wall, cpu = time.perf_counter(), time.process_time()
            try:
                return function(*args, **kwargs)
            finally:
                total = self.totals[name]
                total[0] += time.perf_counter() - wall
                total[1] += time.process_time() - cpu
                total[2] += 1

        return wrapper

    def metrics(self) -> dict:
        """Each phase that ran: its wall and CPU seconds, and how often it ran."""
        return {
            name: {"wall_seconds": w, "cpu_seconds": c, "calls": n}
            for name, (w, c, n) in self.totals.items()
            if n
        }


def machine(versions: dict[str, str], **details) -> dict:
    """The machine the run was on, any `details` of it, and the `versions` of
    the packages the run used."""
    return {
        "platform": platform.platform(),
        "processor": platform.machine(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        **details,
        "versions": versions,
    }


def write_result(result: anndata.AnnData, path: str, metrics: dict) -> None:
    """Saves the run's results, with its measurements in uns["metrics"]."""
    result.uns["metrics"] = json.dumps(metrics)
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(path, compression="gzip")
