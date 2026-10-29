"""Runs st.tl.cci.run once and saves its results and measurements as h5ad.

The acceptance test runs this in a fresh process, so the peak memory is this
run's alone:

    python -m tests.acceptance.run_cci SAMPLE.h5ad RESULT.h5ad [--use-label]

numba's functions are compiled on a small slice first, so the timed run is
steady state. Results are stored sparse, as their difference from the value
most spots have (score 0, p-value 1): the entries of the LR table's sparse
layers, which hold the tested spots, shifted by that value.
"""

import argparse
import contextlib
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
import numba
import numpy
import pandas
import scipy

import stlearn

# Each result, and the value most spots have.
RESULTS = {
    "lr_scores": 0.0,
    "p_vals": 1.0,
    "p_adjs": 1.0,
    "neg_log10_p_adjs": 0.0,
    "lr_sig_scores": 0.0,
}
NEIGHBOURS = ["spot_neighbours", "spot_neigh_bcs"]
# Phases of st.tl.cci.run, timed by wrapping the functions it calls.
PHASES = {
    "neighbours": ("stlearn.tl.cci.analysis", "calc_neighbours"),
    "cell type counts": ("stlearn.tl.cci.analysis", "count"),
    "LR scores": ("stlearn.tl.cci.analysis", "get_lrs_scores"),
    "testing": ("stlearn.tl.cci.analysis", "perform_spot_testing"),
    "testing: backgrounds": ("stlearn.tl.cci.permutation", "score_background"),
}


def shifted(layer, usual: float) -> scipy.sparse.csr_matrix:
    """The layer's values minus `usual`, sparse: 0, left out, where a spot has
    the usual value. From the LR table's sparse layers, whose entries are the
    tested spots and whose other spots have the usual value, or a dense array."""
    if scipy.sparse.issparse(layer):
        out = scipy.sparse.csr_matrix(layer, dtype=numpy.float64, copy=True)
        out.data -= usual
    else:
        out = scipy.sparse.csr_matrix(numpy.asarray(layer, dtype=numpy.float64) - usual)
    out.eliminate_zeros()
    return out


def peak_memory_bytes() -> int:
    """This process's peak resident memory so far."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # Linux: kilobytes


class PhaseTimer:
    """Wall and CPU time of each phase, by wrapping the functions it calls.

    CPU time counts every thread, so CPU / wall is the average number of busy
    cores.
    """

    def __init__(self) -> None:
        self.totals = dict(map(lambda name: (name, [0.0, 0.0, 0]), PHASES))
        self._originals: list[tuple[ModuleType, str, Any]] = []

    def __enter__(self) -> "PhaseTimer":
        for name, (module_name, attribute) in PHASES.items():
            module = importlib.import_module(module_name)
            original = getattr(module, attribute)
            setattr(module, attribute, self._timed(name, original))
            self._originals.append((module, attribute, original))
        return self

    def __exit__(self, *exc) -> None:
        for module, attribute, original in self._originals:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sample")
    parser.add_argument("result")
    parser.add_argument("--use-label", action="store_true")
    parser.add_argument("--n-pairs", type=int, default=1000)
    parser.add_argument("--min-spots", type=int, default=20)
    parser.add_argument("--n-cpus", type=int)
    args = parser.parse_args()

    adata = anndata.read_h5ad(args.sample)
    lrs = stlearn.tl.cci.lr_pairs("connectomeDB2025").names()
    settings = dict(
        min_spots=args.min_spots,
        n_pairs=args.n_pairs,
        n_cpus=args.n_cpus,
        use_label="cell_type" if args.use_label else None,
        random_state=0,
    )

    wall = time.perf_counter()
    warm_up = adata[: min(2000, adata.n_obs)].copy()
    with contextlib.redirect_stdout(sys.stderr):
        stlearn.tl.cci.run(
            warm_up,
            lrs[:60],
            **{**settings, "min_spots": 5, "n_pairs": 200},
            verbose=False,
        )
    warm_up_seconds = time.perf_counter() - wall
    del warm_up

    memory_before = peak_memory_bytes()
    with PhaseTimer() as phases, contextlib.redirect_stdout(sys.stderr):
        wall, cpu = time.perf_counter(), time.process_time()
        lr_results = stlearn.tl.cci.run(adata, lrs, **settings, verbose=False)
        wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    memory_after = peak_memory_bytes()

    threading_layer = None
    with contextlib.suppress(ValueError):  # known only once numba has run
        threading_layer = numba.threading_layer()
    metrics = {
        "bins": adata.n_obs,
        "genes": adata.n_vars,
        "lr_pairs_requested": len(lrs),
        "lr_pairs_tested": int(lr_results.n_vars),
        "settings": settings,
        "warm_up_seconds": warm_up_seconds,
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "peak_memory_bytes": memory_after,
        "memory_added_by_run_bytes": memory_after - memory_before,
        "phases": {
            name: {"wall_seconds": w, "cpu_seconds": c, "calls": n}
            for name, (w, c, n) in phases.totals.items()
            if n
        },
        "machine": {
            "platform": platform.platform(),
            "processor": platform.machine(),
            "python": platform.python_version(),
            "cpus": os.cpu_count(),
            "numba_threads": numba.get_num_threads(),
            "numba_threading_layer": threading_layer,
            "versions": {
                name: importlib.import_module(name).__version__
                for name in ["numpy", "scipy", "pandas", "anndata", "numba"]
            },
        },
    }

    result = anndata.AnnData(obs=pandas.DataFrame(index=adata.obs_names))
    for key, usual in RESULTS.items():
        result.obsm[key] = shifted(lr_results.layers[key], usual)
    for key in NEIGHBOURS:
        result.obs[key] = adata.obsm[key].iloc[:, 0].to_numpy()
    if args.use_label:
        result.obs["cci_het"] = numpy.asarray(adata.obsm["cci_het"]).ravel()
    # The layout of baselines written before the LR table.
    result.uns["lr_summary"] = lr_results.var
    result.uns["lrfeatures"] = lr_results.uns["lrfeatures"]
    result.uns["metrics"] = json.dumps(metrics)
    pathlib.Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(args.result, compression="gzip")


if __name__ == "__main__":
    main()
