"""st.tl.cci.run returns an LR table: the obs/spots/cells as rows and LR pairs as
columns with the most significant first. Its var is the LR summary, each of LR_RESULTS
is a layer, and its uns holds lrfeatures.

The layers are CSC matrices with one structure: an entry for each spot tested for
an LR, those with a score above 0 for it. Elsewhere a result has its default, a
score of 0 or a p-value of 1, which lr_result fills in.

st.tl.cci.run_cci returns cell type interactions and st.tl.cci.run_lr_go's GO terms. The
table it ran on keeps only what is per spot: the neighbours and cell type counts.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from anndata import AnnData
from numba import njit
from scipy import sparse
from statsmodels.stats.multitest import multipletests

LR_RESULTS = ("lr_scores", "p_vals", "p_adjs", "neg_log10_p_adjs", "lr_sig_scores")
DEFAULTS = {"p_vals": 1.0, "p_adjs": 1.0}  # a result's value where an LR was not tested
Matrix = sparse.spmatrix | np.ndarray


def check_lr_results(lr_results: AnnData | None, adata: AnnData | None = None):
    """`lr_results`, if it is the LR table st.tl.cci.run returned (for `adata`'s
    spots, in the same order, when `adata` is given)."""
    if lr_results is None or "lr_scores" not in lr_results.layers:
        raise ValueError("lr_results must be the LR table st.tl.cci.run returned.")
    if adata is not None and not adata.obs_names.equals(lr_results.obs_names):
        raise ValueError(
            "lr_results must have adata's spots, in the same order: the LR table "
            "st.tl.cci.run returned for adata."
        )
    return lr_results


def result_matrix(lr_results: AnnData, key: str) -> Matrix:
    """A result as stored, spots by LR pairs; `neg_log10_p_adjs` from `p_adjs`
    when it is not stored."""
    if key not in LR_RESULTS:
        raise KeyError(f"key must be one of {LR_RESULTS}, not {key!r}")
    check_lr_results(lr_results)
    if key == "neg_log10_p_adjs" and key not in lr_results.layers:
        padjs = lr_results.layers["p_adjs"]
        with np.errstate(divide="ignore"):
            if sparse.issparse(padjs):
                padjs = sparse.csc_matrix(padjs)
                return _with_data(padjs, -np.log10(padjs.data))
            return -np.log10(np.asarray(padjs))
    return lr_results.layers[key]


def lr_result(lr_results: AnnData, key: str, lr: str | int | None = None) -> np.ndarray:
    """A result dense, spots by LR pairs, or one LR pair's values with `lr` (its
    name or position); untested spots get the result's default. The whole matrix
    of a large sample is big, so ask for one LR at a time."""
    matrix = result_matrix(lr_results, key)
    if lr is None:
        return to_dense(matrix, DEFAULTS.get(key, 0.0))
    j = lr if isinstance(lr, int | np.integer) else lr_results.var_names.get_loc(lr)
    return to_dense(matrix[:, [j]], DEFAULTS.get(key, 0.0)).ravel()


def to_dense(matrix: Matrix, default: float = 0.0) -> np.ndarray:
    """`matrix` dense, with `default` where a sparse matrix has no entry."""
    if not sparse.issparse(matrix):
        return np.asarray(matrix)
    coo = sparse.coo_matrix(matrix)
    dense = np.full(coo.shape, default)
    dense[coo.row, coo.col] = coo.data
    return dense


def scored_entries(lr_scores: Matrix) -> sparse.csc_matrix:
    """The scores above 0 as a canonical CSC matrix: the structure every result
    shares. From the sparse scores the pipeline makes or a dense array."""
    scores = sparse.csc_matrix(lr_scores, copy=True)
    scores.data[~(scores.data > 0)] = 0
    scores.eliminate_zeros()
    scores.sum_duplicates()
    return scores


def _with_data(structure: sparse.csc_matrix, data: np.ndarray) -> sparse.csc_matrix:
    """A CSC matrix with `structure`'s entries and these values; its own indices,
    so no layer shares them."""
    return sparse.csc_matrix(
        (data, structure.indices.copy(), structure.indptr.copy()), shape=structure.shape
    )


def on_structure(values: Matrix, structure: sparse.csc_matrix) -> sparse.csc_matrix:
    """`values`, dense or sparse with the same entries, on `structure`'s entries."""
    if not sparse.issparse(values):
        columns = np.repeat(np.arange(structure.shape[1]), np.diff(structure.indptr))
        data = np.asarray(values)[structure.indices, columns]
        return _with_data(structure, data.astype(np.float64))
    values = sparse.csc_matrix(values)
    values.sum_duplicates()
    if not (
        np.array_equal(values.indptr, structure.indptr)
        and np.array_equal(values.indices, structure.indices)
    ):
        raise ValueError("The matrices must have entries for the same spots and LRs.")
    return _with_data(structure, values.data.astype(np.float64))


@njit(cache=True)
def _fdr_bh(indptr: np.ndarray, pvals: np.ndarray, out: np.ndarray) -> None:
    """Benjamini-Hochberg within each group pvals[indptr[i] : indptr[i + 1]], into
    out: statsmodels' multipletests(method="fdr_bh") operation for operation."""
    for i in range(len(indptr) - 1):
        start, end = indptr[i], indptr[i + 1]
        n = end - start
        if n == 0:
            continue
        group = pvals[start:end]
        order = np.argsort(group)
        adjusted = np.empty(n, np.float64)
        for k in range(n):
            adjusted[k] = group[order[k]] / ((k + 1) / n)
        for k in range(n - 2, -1, -1):
            if adjusted[k + 1] < adjusted[k]:
                adjusted[k] = adjusted[k + 1]
        for k in range(n):
            if adjusted[k] > 1:
                adjusted[k] = 1.0
            out[start + order[k]] = adjusted[k]


def correct_pvals(
    lr_scores: Matrix,
    pvals: Matrix,
    axis: str | None = "spot",
    method: str = "fdr_bh",
) -> Matrix:
    """Adjusted p-values, spots by LR pairs: across the LRs tested in each spot
    ('spot'), the spots tested for each LR ('LR'), or not at all (None). A spot-LR
    pair was tested if it has a score above 0; the rest are 1. Sparse, on the
    tested entries, when either input is."""
    if axis is None:
        return pvals.copy()
    if axis not in ("spot", "LR"):
        raise ValueError(
            "Invalid correct_axis input, must be one of: 'LR', 'spot', or None",
        )
    ps = on_structure(pvals, scored_entries(lr_scores))
    groups = ps.tocsr() if axis == "spot" else ps  # a group's entries contiguous
    adjusted = np.ones_like(groups.data)
    if method == "fdr_bh":
        _fdr_bh(groups.indptr, groups.data, adjusted)
    else:
        for start, end in zip(groups.indptr[:-1], groups.indptr[1:], strict=True):
            if end > start:
                adjusted[start:end] = multipletests(
                    groups.data[start:end], method=method
                )[1]
    padjs = type(groups)((adjusted, groups.indices, groups.indptr), groups.shape)
    padjs = padjs.tocsc()
    if sparse.issparse(lr_scores) or sparse.issparse(pvals):
        return padjs
    return to_dense(padjs, 1.0)


def store_lr_results(
    lr_results: AnnData,
    lr_scores: Matrix,
    pvals: Matrix,
    padjs: Matrix,
    pval_adj_cutoff: float,
) -> None:
    """Writes the LR summary to the LR table's var and the LR_RESULTS layers, CSC
    on the tested entries, with LRs ordered by their number of significant spots.

    The inputs' columns follow `lr_results.var_names` on the way in. Previous
    results are replaced; the rest of the table (X, varm, other layers) moves with
    the summary, as scanpy does. 'source', where each pair came from, is kept; any
    other summary columns, such as run_cci's, are dropped: they describe the
    previous results.
    """
    scores = scored_entries(lr_scores)
    adj, ps = on_structure(padjs, scores), on_structure(pvals, scores)
    n_lrs = scores.shape[1]
    column = np.repeat(np.arange(n_lrs), np.diff(scores.indptr))
    astype = np.diff(scores.indptr).astype(np.int64)
    n_spots_sig = np.bincount(column[adj.data < pval_adj_cutoff], minlength=n_lrs)
    n_spots_sig_pvals = np.bincount(column[ps.data < pval_adj_cutoff], minlength=n_lrs)
    summary = pd.DataFrame(
        {
            "n_spots": astype,
            "n_spots_sig": n_spots_sig,
            "n_spots_sig_pval": n_spots_sig_pvals,
        },
        index=lr_results.var_names,
    )
    if "source" in lr_results.var:
        summary.insert(0, "source", lr_results.var["source"])
    order: npt.NDArray[np.intp] = np.argsort(-summary["n_spots_sig"].to_numpy())

    for key in LR_RESULTS:
        lr_results.layers.pop(key, None)
    lr_results.var = summary
    lr_results._inplace_subset_var(order)
    scores, ps, adj = scores[:, order], ps[:, order], adj[:, order]
    sig = adj.data < pval_adj_cutoff
    lr_results.layers["lr_scores"] = scores
    lr_results.layers["p_vals"] = ps
    lr_results.layers["p_adjs"] = adj
    with np.errstate(divide="ignore"):
        lr_results.layers["neg_log10_p_adjs"] = _with_data(adj, -np.log10(adj.data))
    lr_results.layers["lr_sig_scores"] = _with_data(
        scores, np.where(sig, scores.data, 0.0)
    )
