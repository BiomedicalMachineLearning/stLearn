"""Building each LR pair's background from random gene pairs of similar
expression."""

import numpy as np
import numpy.typing as npt
import pandas as pd
from anndata import AnnData
from numba import njit, prange
from scipy import sparse


def nonzero_quantile(expr, q, method):
    """Calculating the non-zero quantiles."""
    nonzero_expr = expr[expr > 0]
    quants = np.quantile(nonzero_expr, q=q, method=method)
    if not isinstance(quants, np.ndarray) or quants.ndim == 0:
        quants = np.array([quants])
    return quants


def nonnegative(X: sparse.spmatrix) -> bool:
    """Whether every stored value of X is >= 0 (so not NaN either); so too with
    nothing stored."""
    return X.nnz == 0 or bool(X.data.min() >= 0)


def sparse_quantiles(X: sparse.spmatrix, quantiles) -> np.ndarray:
    """np.quantile(X.toarray(), quantiles, axis=0, method="nearest") for a sparse
    matrix with no negative values, from its stored values: quantiles (rows) of
    each column.

    A column's zeros sort first, so quantile q is 0 unless the order statistic
    np.quantile picks lies among the column's stored values, and then it is that
    one of them sorted. np.quantile itself says which order statistic, so the
    rounding is the installed numpy's.
    """
    X = sparse.csc_matrix(X)  # column by column; no copy when already CSC
    quantiles = np.atleast_1d(quantiles)
    n, m = X.shape
    order = np.quantile(np.arange(n), quantiles, method="nearest").astype(np.int64)
    data = X.data.copy()  # never sort X.data itself: it is aligned with X.indices
    for start, end in zip(X.indptr[:-1], X.indptr[1:], strict=True):
        data[start:end].sort()
    position = order[:, None] - (n - np.diff(X.indptr))  # (q, m); < 0 => a zero
    hit = position >= 0
    out = np.zeros((len(order), m), dtype=X.dtype)
    out[hit] = data[(X.indptr[:-1] + position)[hit]]
    return out


def _column_quantiles(expr, quantiles: np.ndarray, nonzero: bool) -> np.ndarray:
    """np.quantile(..., method="nearest") of each column of expr (spots by genes):
    quantiles as rows, genes as columns. With `nonzero`, of the column's values
    above 0 only. Sparse expr is done from its stored values."""
    if not sparse.issparse(expr):
        # a plain array: anndata's views trip up numpy
        expr = np.asarray(expr)
        if not nonzero:
            return np.quantile(expr, quantiles, axis=0, method="nearest")
        return np.apply_along_axis(
            nonzero_quantile, 0, expr, q=quantiles, method="nearest"
        )
    expr = sparse.csc_matrix(expr)
    if nonzero:  # only the stored values can be above 0
        per_column = [
            nonzero_quantile(
                expr.data[expr.indptr[j] : expr.indptr[j + 1]], quantiles, "nearest"
            )
            for j in range(expr.shape[1])
        ]
        # The first column's dtype, as np.apply_along_axis gives.
        return np.stack(per_column, axis=1).astype(per_column[0].dtype, copy=False)
    if not nonnegative(expr):  # negative values sort before the zeros
        return _column_quantiles(expr.toarray(), quantiles, nonzero)
    return sparse_quantiles(expr, quantiles)


def get_lr_quants(
    lr_expr,
    l_indices: list,
    r_indices: list,
    quantiles: np.ndarray,
    quant_method="",
):
    """Gets the quantiles per gene in the LR pair, & then concatenates.

    lr_expr is spots by genes: an array or a sparse matrix.
    Returns
    -------
    lr_quants:
        Each row is a quantile value, each column is an LR pair.
    """

    # First getting the quantiles of gene expression #
    gene_quants = _column_quantiles(
        lr_expr, quantiles, nonzero=quant_method != "quantiles"
    )

    l_quants = gene_quants[:, l_indices]
    r_quants = gene_quants[:, r_indices]
    lr_quants = np.concatenate((l_quants, r_quants), 0).transpose()

    return lr_quants


def get_lr_zeroprops(lr_expr, l_indices: list, r_indices: list):
    """Gets the proportion of zeros per gene in the LR pair, & then concatenates.

    lr_expr is spots by genes: an array or a sparse matrix.
    Returns
    -------
    lr_props:
        Each row is a prop value, each column is an LR pair.
    """

    # First getting the quantiles of gene expression #
    if sparse.issparse(lr_expr):
        stored = sparse.csc_matrix(lr_expr, copy=True)
        stored.eliminate_zeros()  # a stored 0 is a zero
        n_nonzero = np.diff(stored.indptr)
        gene_props = ((lr_expr.shape[0] - n_nonzero) / lr_expr.shape[0])[np.newaxis]
    else:
        gene_props = (lr_expr == 0).mean(axis=0, keepdims=True)

    l_props = gene_props[:, l_indices]
    r_props = gene_props[:, r_indices]
    lr_props = np.concatenate((l_props, r_props), 0).transpose()

    return lr_props


@njit(parallel=True)
def get_similar_genes_fast(
    ref_quants: np.ndarray,
    n_genes: int,
    candidate_quants: np.ndarray,
    candidate_genes: np.ndarray,
) -> np.ndarray:
    """The n_genes candidate genes whose expression quantiles are closest to
    ref_quants, by the sum over quantiles of |ref - cand| / (ref + cand),
    excluding exact matches (likely the same gene)."""
    n_quantiles, n_candidates = candidate_quants.shape
    dists = np.zeros(n_candidates, np.float64)
    for i in prange(n_candidates):
        total = 0.0
        for k in range(n_quantiles):
            ref, cand = ref_quants[k], candidate_quants[k, i]
            term = abs(ref - cand) / (ref + cand)
            if not np.isnan(term):
                total += term
        dists[i] = total
    keep = dists > 0
    order = np.argsort(dists[keep])
    return candidate_genes[keep][order[:n_genes]]


def gen_rand_pairs(
    genes1: npt.NDArray[np.int64],
    genes2: npt.NDArray[np.int64],
    n_pairs: int,
    seed: int,
) -> npt.NDArray[np.int64]:
    """Generate unique random gene pairs for building background LR scores.

    Each pair a combination from genes1 and genes2. Self-pairs and duplicate pairs
    are rejected. Genes are positions in adata.var_names.

    Parameters
    ----------
    genes1: npt.NDArray[np.int64]
        Candidate genes for the ligand position in adata.var_names.
    genes2: npt.NDArray[np.int64]
        Candidate genes for the receptor position in adata.var_names.
    n_pairs: int
        Number of unique pairs to generate.
    seed: int
        Seed for the random generator, for reproducible pair selection.
    Returns
    -------
    pairs: npt.NDArray[np.int64]
        Shape (n_pairs, 2): each pair's (ligand, receptor) positions.
    """
    if np.unique(genes1).size < len(genes1) or np.unique(genes2).size < len(genes2):
        raise ValueError("genes1 and genes2 must each list a gene once.")
    n_possible = len(genes1) * len(genes2) - np.intersect1d(genes1, genes2).size
    if n_pairs > n_possible:
        raise ValueError(
            f"Requested {n_pairs} unique pairs but only {n_possible} are "
            f"possible from {len(genes1)}x{len(genes2)} genes."
        )
    return _gen_rand_pairs(genes1, genes2, n_pairs, seed)


@njit
def _gen_rand_pairs(genes1, genes2, n_pairs, seed):
    np.random.seed(seed)  # noqa: NPY002 (numba requires legacy API)
    pairs = np.empty((n_pairs, 2), np.int64)
    seen = np.zeros((len(genes1), len(genes2)), np.bool_)
    for j in range(n_pairs):
        i1 = np.random.randint(0, len(genes1))  # noqa: NPY002
        i2 = np.random.randint(0, len(genes2))  # noqa: NPY002
        while seen[i1, i2] or genes1[i1] == genes2[i2]:
            i1 = np.random.randint(0, len(genes1))  # noqa: NPY002
            i2 = np.random.randint(0, len(genes2))  # noqa: NPY002
        pairs[j, 0], pairs[j, 1] = genes1[i1], genes2[i2]
        seen[i1, i2] = True
    return pairs


def get_lr_features(lr_expr: pd.DataFrame | AnnData, lrs, quantiles):
    """Gets expression features of LR pairs; nonzero-median, zero-prop, quantiles.

    lr_expr is the LR genes' expression, spots by genes: a DataFrame, or an
    AnnData of those genes, whose sparse X is used as it is, never densified.
    """
    quantiles = np.array(quantiles)
    if isinstance(lr_expr, AnnData):
        genes, expr = lr_expr.var_names.to_numpy(), lr_expr.X
        # Column by column below, so CSC; dense stays as it is.
        expr = sparse.csc_matrix(expr) if sparse.issparse(expr) else np.asarray(expr)
    else:
        genes, expr = lr_expr.columns.values, lr_expr.values

    # Determining indices of LR pairs #
    l_indices, r_indices = [], []
    for lr in lrs:
        l_, r_ = lr.split("_")
        l_indices.extend(np.where(genes == l_)[0])
        r_indices.extend(np.where(genes == r_)[0])

    # The nonzero median when quantiles=.5 #
    lr_quants = get_lr_quants(
        expr, l_indices, r_indices, quantiles, quant_method="quantiles"
    )

    # Calculating the zero proportions for grouping based on median/zeros #
    lr_props = get_lr_zeroprops(expr, l_indices, r_indices)

    # Getting lr features for later diagnostics
    lr_meds = get_lr_quants(
        expr, l_indices, r_indices, quantiles=np.array([0.5]), quant_method=""
    )
    lr_median_means = lr_meds.mean(axis=1)
    lr_prop_means = lr_props.mean(axis=1)

    # Calculating mean rank #
    median_order = np.argsort(lr_median_means)
    prop_order = np.argsort(lr_prop_means * -1)
    median_ranks = [np.where(median_order == i)[0][0] for i in range(len(lrs))]
    prop_ranks = [np.where(prop_order == i)[0][0] for i in range(len(lrs))]
    mean_ranks = np.array([median_ranks, prop_ranks]).mean(axis=0)

    # Saving the lrfeatures...
    cols = ["nonzero-median", "zero-prop", "median_rank", "prop_rank", "mean_rank"]
    lr_features_data = {
        cols[0]: np.array(lr_median_means, dtype=np.float64),
        cols[1]: np.array(lr_prop_means, dtype=np.float64),
        cols[2]: np.array(median_ranks, dtype=np.float64),
        cols[3]: np.array(prop_ranks, dtype=np.float64),
        cols[4]: np.array(mean_ranks, dtype=np.float64),
    }
    lr_features = pd.DataFrame(lr_features_data, index=lrs)
    lr_features = lr_features.iloc[np.argsort(mean_ranks), :]
    lr_cols = [f"L_{quant}" for quant in quantiles] + [
        f"R_{quant}" for quant in quantiles
    ]
    quant_df = pd.DataFrame(lr_quants, columns=lr_cols, index=lrs, dtype=np.float64)
    lr_features = pd.concat((lr_features, quant_df), axis=1)

    return lr_features


def background_pairs(
    lr_: str,
    l_quant: np.ndarray,
    r_quant: np.ndarray,
    candidates: npt.NDArray[np.int64],
    candidate_quants: np.ndarray,
    gene_bg_genes: dict[str, np.ndarray],
    n_genes: int,
    n_pairs: int,
    seed: int,
) -> npt.NDArray[np.int64]:
    """The random gene pairs of an LR's background: n_pairs pairs of the n_genes
    candidates most like its ligand and its receptor in expression quantiles.
    `candidates` are positions in adata.var_names; `gene_bg_genes` caches each
    gene's similar genes across LRs."""
    similar = []
    for gene, quant in ((lr_.split("_")[0], l_quant), (lr_.split("_")[1], r_quant)):
        if gene not in gene_bg_genes:
            gene_bg_genes[gene] = get_similar_genes_fast(
                quant, n_genes, candidate_quants, candidates
            )
        similar.append(gene_bg_genes[gene])
    return gen_rand_pairs(similar[0], similar[1], n_pairs, seed)
