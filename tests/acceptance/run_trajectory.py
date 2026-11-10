"""Runs st.spatial.trajectory.pseudotime and pseudotimespace_global once and
saves their results and measurements as h5ad.

The acceptance test runs this in a fresh process, so the peak memory is this
run's alone:

    python -m tests.acceptance.run_trajectory SAMPLE.h5ad RESULT.h5ad \
        [--lineage 4,5,6] [--all-lineages]

pseudotime runs on a few clusters first, so the timed run is steady state.
--all-lineages then runs pseudotimespace_global on every path pseudotime
stored, as the report in issue #340 did, and counts how each one ends.
"""

import argparse
import contextlib
import importlib.metadata
import io
import json
import re
import sys
import time
from collections import Counter

import anndata
import numpy
import pandas

import stlearn
from stlearn.utils import _read_graph
from tests.acceptance.measure import (
    PhaseTimer,
    machine,
    peak_memory_bytes,
    write_result,
)

# Phases of the run, timed by wrapping the functions it calls.
PHASES = {
    "subclusters": ("stlearn.spatial.trajectory.pseudotime", "localization"),
    "PAGA": ("scanpy.tl", "paga"),
    "diffusion map": ("scanpy.tl", "diffmap"),
    "DPT": ("scanpy.tl", "dpt"),
    "available paths": (
        "stlearn.spatial.trajectory.pseudotime",
        "store_available_paths",
    ),
    "global level": ("stlearn.spatial.trajectory.pseudotimespace", "global_level"),
}
LABEL = "cluster"


def first_clusters(adata: anndata.AnnData, n: int) -> anndata.AnnData:
    """The bins of the first `n` clusters, for the warm-up."""
    keep = adata.obs[LABEL].cat.categories[:n]
    small = adata[adata.obs[LABEL].isin(keep)].copy()
    small.obs[LABEL] = small.obs[LABEL].cat.remove_unused_categories()
    small.uns["iroot"] = int(
        numpy.argmin(small.obs["imagerow"] + small.obs["imagecol"])
    )
    return small


def outcome_of(error: Exception | None) -> str:
    """How a pseudotimespace_global call ended, without the cluster names."""
    if error is None:
        return "OK"
    return f"{type(error).__name__}: " + re.sub(r"\[[^\]]*\]", "[...]", str(error))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sample")
    parser.add_argument("result")
    parser.add_argument("--eps", type=float, default=7.0)
    parser.add_argument("--max-nodes", type=int, default=4)
    parser.add_argument("--lineage", default="4,5,6")
    parser.add_argument("--all-lineages", action="store_true")
    args = parser.parse_args()

    adata = anndata.read_h5ad(args.sample)
    lineage = args.lineage.split(",")
    settings = dict(use_label=LABEL, eps=args.eps, max_nodes=args.max_nodes)

    wall = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        stlearn.spatial.trajectory.pseudotime(first_clusters(adata, 3), **settings)
    warm_up_seconds = time.perf_counter() - wall

    memory_before = peak_memory_bytes()
    lineages: Counter[str] = Counter()
    with PhaseTimer(PHASES) as phases, contextlib.redirect_stdout(sys.stderr):
        wall, cpu = time.perf_counter(), time.process_time()
        stlearn.spatial.trajectory.pseudotime(adata, **settings)
        pseudotime_seconds = time.perf_counter() - wall

        global_wall = time.perf_counter()
        stlearn.spatial.trajectory.pseudotimespace_global(
            adata, use_label=LABEL, list_clusters=lineage
        )
        global_seconds = time.perf_counter() - global_wall
        pts_graph = _read_graph(adata, "PTS_graph")

        if args.all_lineages:
            # As the report in #340 did. Each call overwrites PTS_graph.
            for path in adata.uns["available_paths"].values():
                error = None
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        stlearn.spatial.trajectory.pseudotimespace_global(
                            adata, use_label=LABEL, list_clusters=list(path)
                        )
                except Exception as e:
                    error = e
                lineages[outcome_of(error)] += 1
        wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    memory_after = peak_memory_bytes()

    paths = sorted(
        list(map(str, path)) for path in adata.uns["available_paths"].values()
    )
    metrics = {
        "bins": adata.n_obs,
        "clusters": len(adata.obs[LABEL].cat.categories),
        "subclusters_kept": sum(len(v) for v in adata.uns["split_node"].values()),
        "available_paths": len(paths),
        "settings": {**settings, "lineage": lineage, "all_lineages": args.all_lineages},
        "warm_up_seconds": warm_up_seconds,
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "pseudotime_seconds": pseudotime_seconds,
        "pseudotimespace_global_seconds": global_seconds,
        "peak_memory_bytes": memory_after,
        "memory_added_by_run_bytes": memory_after - memory_before,
        "lineage_outcomes": dict(lineages),
        "phases": phases.metrics(),
        "machine": machine(
            {
                name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "anndata", "scanpy", "networkx"]
            }
        ),
    }

    result = anndata.AnnData(obs=pandas.DataFrame(index=adata.obs_names))
    for key in ["sub_cluster_labels", "dpt_pseudotime"]:
        result.obs[key] = adata.obs[key].to_numpy()
    # Paths are compared as a set: their keys number every path found,
    # including those too long to keep, so a faster search renumbers them.
    result.uns["available_paths"] = json.dumps(paths)
    result.uns["split_node"] = json.dumps(
        {k: sorted(map(str, v)) for k, v in adata.uns["split_node"].items()}
    )
    result.uns["pts_graph"] = json.dumps(
        sorted([str(u), str(v), float(w)] for u, v, w in pts_graph.edges(data="weight"))
    )
    write_result(result, args.result, metrics)


if __name__ == "__main__":
    main()
