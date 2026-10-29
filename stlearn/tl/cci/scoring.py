"""LR scores: each spot's ligand-receptor co-expression with its neighbours.

A score needs each gene's mean over the spot's neighbours. A gene is in many
pairs, so the means are computed once per gene (neighbourhood_means) and the
pairs then only look them up (score_pairs). LRScorer prepares a sample once for
repeated scoring, as perform_spot_testing does thousands of times.

Spots are scored a chunk at a time, with only the rows a chunk needs (its spots
and their neighbours) dense, so memory does not grow with the sample. With
sparse_result the scores come back as a spots-by-pairs CSC matrix of the scores
above 0: the tested ones, a small fraction for a large sample.
"""

import numpy as np
import numpy.typing as npt
from anndata import AnnData
from numba import njit, prange
from numba.typed import List
from scipy import sparse

MEANS_CHUNK_BYTES = 2**28  # memory for a chunk's neighbourhood means and scores
Expression = sparse.spmatrix | np.ndarray


class LRScorer:
    """Scores LR pairs on a sample, with the sample prepared once for every call:
    the neighbours flattened, and sparse expression kept by column so a call
    takes out just its genes, in time proportional to their values rather than
    to the whole matrix.

    Parameters
    ----------
    adata: AnnData
        See get_lrs_scores.
    neighbours: List
        Array of arrays with indices specifying neighbours of each spot.
    by_column: bool
        Keep a column-oriented copy of sparse expression, the size of adata.X.
        Worth it for repeated calls; get_lrs_scores passes False for its one.
    """

    def __init__(self, adata: AnnData, neighbours: List, by_column: bool = True):
        self.adata = adata
        self.adjacency = adjacency(neighbours)
        self.columns: sparse.csc_matrix | None = (
            sparse.csc_matrix(adata.X)
            if by_column and sparse.issparse(adata.X)
            else None
        )
        """adata.X by column, for taking out genes; None for dense data or
        without by_column."""

    def expression(self, genes: npt.NDArray[np.int64]) -> Expression:
        """Spots by `genes` (positions in adata.var_names): sparse data as CSR,
        so a chunk of spots' rows are cheap to take out; dense data as it is."""
        if self.columns is not None:
            return self.columns[:, genes].tocsr()
        expr = self.adata[:, genes].X
        if sparse.issparse(expr):
            return sparse.csr_matrix(expr)
        return np.asfortranarray(expr)

    def score(
        self,
        lrs: npt.NDArray[np.str_],
        het_vals: np.ndarray,
        min_expr: float,
        filter_pairs: bool = True,
        spot_indices: npt.NDArray[np.int32] | None = None,
        sparse_result: bool = False,
    ) -> tuple[npt.NDArray[np.float64] | sparse.csc_matrix, npt.NDArray[np.str_]]:
        """The scores of the LR pairs, named 'L_R'; see get_lrs_scores."""
        names = np.array(list(map(lambda lr: lr.split("_"), lrs))).reshape(-1, 2)
        pairs = self.adata.var_names.get_indexer(names.ravel()).reshape(names.shape)
        found = (pairs >= 0).all(axis=1)
        if filter_pairs:
            names, pairs = names[found], pairs[found]
        elif not found.all():
            raise KeyError(
                f"Genes not in adata.var_names: {np.unique(names[pairs < 0]).tolist()}"
            )
        lr_scores = self.score_indices(
            pairs, het_vals, min_expr, spot_indices, sparse_result
        )
        new_lrs = np.array(list(map("_".join, names)), dtype=str)
        return lr_scores, new_lrs

    def score_indices(
        self,
        pairs: npt.NDArray[np.int64],
        het_vals: np.ndarray,
        min_expr: float,
        spot_indices: npt.NDArray[np.int32] | None = None,
        sparse_result: bool = False,
    ) -> npt.NDArray[np.float64] | sparse.csc_matrix:
        """The scores of pairs given as positions in adata.var_names, so the
        thousands of random pairs of a background are never named.

        Parameters
        ----------
        pairs: npt.NDArray[np.int64]
            Pairs*2 positions in adata.var_names, as (ligand, receptor).
        het_vals, min_expr, spot_indices, sparse_result
            See get_lrs_scores.
        Returns
        -------
        lr_scores: npt.NDArray[np.float64] | scipy.sparse.csc_matrix
            Shape (n_scored_spots, n_pairs), as get_lrs_scores returns.
        """
        if spot_indices is None:
            spot_indices = np.arange(len(self.adata), dtype=np.int32)
        # Only the genes in the pairs; each pair is two columns of `expr`.
        genes, columns = np.unique(pairs, return_inverse=True)
        expr = self.expression(genes)
        return score_spots(
            expr,
            columns.reshape(pairs.shape),
            self.adjacency,
            het_vals,
            min_expr,
            spot_indices,
            sparse_result,
        )


def get_lrs_scores(
    adata: AnnData,
    lrs: npt.NDArray[np.str_],
    neighbours: np.ndarray,
    het_vals: np.ndarray,
    min_expr: float,
    filter_pairs: bool = True,
    spot_indices: npt.NDArray[np.int32] | None = None,
    sparse_result: bool = False,
) -> tuple[npt.NDArray[np.float64] | sparse.csc_matrix, npt.NDArray[np.str_]]:
    """Gets the scores for the indicated set of LR pairs & the heterogeneity values.

    For repeated calls on the same sample, make an LRScorer once and call its
    score instead.

    Parameters
    ----------
    adata: AnnData
        See run() doc-string.
    lrs: np.ndarray
        See run() doc-string.
    neighbours: np.ndarray
        Array of arrays with indices specifying neighbours of each spot.
    het_vals: np.ndarray
        Cell heterogeneity counts per spot.
    min_expr: float
        Minimum gene expression of either L or R for spot to be considered to
        have reasonable score.
    filter_pairs: bool
        Whether to filter to valid pairs or not.
    spot_indices: npt.NDArray[np.int32]
        Subset of spots to score, given as their integer row positions.
    sparse_result: bool
        Return the scores as a scipy.sparse.csc_matrix holding only the scores
        above 0, the spots that are tested for each LR. The matrix of every spot
        by every LR pair of a large sample is otherwise far too big to hold.
    Returns
    -------
    lr_scores: npt.NDArray[np.float64] | scipy.sparse.csc_matrix
        Shape (n_scored_spots, n_pairs). LR score for each scored spot (rows in
        the order of spot_indices) and each ligand-receptor pair (columns).
        Sparse with sparse_result, holding the scores above 0.
    new_lrs: npt.NDArray[np.str_]
        Shape (n_pairs,). Ligand-receptor pair labels in 'L_R' format
        (e.g. 'L1_R1'), column-aligned with lr_scores.
    """
    scorer = LRScorer(adata, neighbours, by_column=False)
    return scorer.score(
        lrs, het_vals, min_expr, filter_pairs, spot_indices, sparse_result
    )


@njit(cache=True)
def direction_score(at_spot: float, mean_partner: float, min_expr: float) -> float:
    """One direction of an LR score at a spot: a gene's expression at the spot
    with its partner gene's mean expression over the spot's neighbours.

    Parameters
    ----------
    at_spot: float
        Expression of the gene at the spot itself.
    mean_partner: float
        Mean expression of the partner gene over the spot's neighbours.
    min_expr: float
        Minimum expression for gene to be considered expressed.
    Returns
    -------
    score: float
        The score for this direction at this spot.
    """
    return at_spot * (mean_partner > min_expr) + (at_spot > min_expr) * mean_partner


@njit(cache=True)
def flatten_neighbours(neighbours: List) -> tuple[np.ndarray, np.ndarray]:
    """The neighbours as one array, spot i's at flat[starts[i] : starts[i + 1]]."""
    starts = np.zeros(len(neighbours) + 1, np.int64)
    for i in range(len(neighbours)):
        starts[i + 1] = starts[i] + len(neighbours[i])
    flat = np.empty(starts[-1], np.int64)
    for i in range(len(neighbours)):
        flat[starts[i] : starts[i + 1]] = neighbours[i]
    return starts, flat


def adjacency(neighbours: List) -> sparse.csr_matrix:
    """The neighbours as a spots-by-spots CSR matrix: row i's indices are spot
    i's neighbours, in their order, so rows of spots slice out cheaply."""
    starts, flat = flatten_neighbours(neighbours)
    n = len(starts) - 1
    return sparse.csr_matrix((np.ones(len(flat), np.int8), flat, starts), shape=(n, n))


@njit(parallel=True)
def neighbourhood_means(
    expr: np.ndarray,
    starts: np.ndarray,
    flat: np.ndarray,
    spot_indices: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """Each gene's mean expression over each scored spot's neighbours; 0 for a
    spot with no neighbours. Computed once per gene, however many pairs it is in.

    Parameters
    ----------
    expr: np.ndarray
        Spots*Genes expression, column-major.
    starts, flat: np.ndarray
        The neighbours from flatten_neighbours, as rows of expr.
    spot_indices: npt.NDArray[np.int32]
        The spots to compute the means at, as positions in starts.
    Returns
    -------
    means: npt.NDArray[np.float64]
        Genes*Scored spots, so means[g, s] is at spot spot_indices[s].
    """
    means = np.zeros((expr.shape[1], len(spot_indices)), np.float64)
    for g in prange(expr.shape[1]):
        gene = expr[:, g]
        for s in range(len(spot_indices)):
            start, end = starts[spot_indices[s]], starts[spot_indices[s] + 1]
            if end > start:
                total = 0.0
                for k in range(start, end):
                    total += gene[flat[k]]
                means[g, s] = total / (end - start)
    return means


@njit(parallel=True)
def score_pairs(
    expr: np.ndarray,
    means: np.ndarray,
    pairs: np.ndarray,
    het_vals: np.ndarray,
    min_expr: float,
    spot_indices: npt.NDArray[np.int32],
    spot_scores: npt.NDArray[np.float64],
) -> None:
    """Fills spot_scores with the scores from the genes' expression and
    neighbourhood means.

    Parameters
    ----------
    expr: np.ndarray
        Spots*Genes expression, column-major.
    means: np.ndarray
        Genes*Scored spots neighbourhood means, from neighbourhood_means.
    pairs: np.ndarray
        Pairs*2 column indices into expr, as (ligand, receptor).
    het_vals: np.ndarray
        Spots*Het counts, a value per row of expr.
    min_expr: float
        Minimum expression for gene to be considered expressed.
    spot_indices: npt.NDArray[np.int32]
        The scored spots, as rows of expr.
    spot_scores: npt.NDArray[np.float64]
        Pairs*Scored spots, written; each thread writes one pair's row.
    """
    for i in prange(pairs.shape[0]):
        ligand, receptor = pairs[i, 0], pairs[i, 1]
        for s in range(len(spot_indices)):
            si = spot_indices[s]
            score = (
                direction_score(expr[si, ligand], means[receptor, s], min_expr)
                + direction_score(expr[si, receptor], means[ligand, s], min_expr)
            ) / 2.0
            # The merge scores #
            spot_scores[i, s] = het_vals[si] * score


def spots_per_chunk(n_genes: int, n_pairs: int) -> int:
    """Spots scored at a time: as many as keep their means (genes by spots) and
    scores (pairs by spots), float64, within MEANS_CHUNK_BYTES."""
    return max(1, MEANS_CHUNK_BYTES // max(1, 8 * (n_genes + n_pairs)))


def _dense_rows(expr: Expression, rows: np.ndarray) -> np.ndarray:
    """These rows of expr, dense and column-major."""
    taken = expr[rows]
    return (
        taken.toarray(order="F") if sparse.issparse(taken) else np.asfortranarray(taken)
    )


def score_spots(
    expr: Expression,
    pairs: np.ndarray,
    adjacency: sparse.csr_matrix,
    het_vals: np.ndarray,
    min_expr: float,
    spot_indices: npt.NDArray[np.int32],
    sparse_result: bool = False,
) -> npt.NDArray[np.float64] | sparse.csc_matrix:
    """Scores the pairs at the spots, a chunk of spots at a time, each chunk's
    rows of expr dense and its neighbourhood means held only while it is scored.
    See get_scores for the arguments and result; with sparse_result the scores
    come back as get_lrs_scores describes.
    """
    het_vals = np.asarray(het_vals)
    n_pairs, n_spots = pairs.shape[0], len(spot_indices)
    chunk = spots_per_chunk(expr.shape[1], n_pairs)
    pieces: list[sparse.csr_matrix] = []
    spot_scores = np.empty((n_pairs, 0 if sparse_result else n_spots), np.float64)
    for start in range(0, n_spots, chunk):
        spots = spot_indices[start : start + chunk]
        neighbours = adjacency[spots]
        rows, local = np.unique(
            np.concatenate([spots, neighbours.indices]), return_inverse=True
        )
        block = _dense_rows(expr, rows)
        means = neighbourhood_means(
            block,
            neighbours.indptr.astype(np.int64),
            local[len(spots) :].astype(np.int64),
            np.arange(len(spots), dtype=np.int32),
        )
        scores = (
            np.empty((n_pairs, len(spots)), np.float64)
            if sparse_result
            else spot_scores[:, start : start + chunk]
        )
        block_spots = local[: len(spots)].astype(np.int32)
        score_pairs(block, means, pairs, het_vals[rows], min_expr, block_spots, scores)
        if sparse_result:
            by_spot = scores.T
            above = by_spot > 0
            pieces.append(
                sparse.csr_matrix((by_spot[above], np.nonzero(above)), by_spot.shape)
            )
            del scores, by_spot, above  # before the next chunk's are made
    if not sparse_result:
        return spot_scores.T
    if not pieces:
        return sparse.csc_matrix((n_spots, n_pairs), dtype=np.float64)
    return sparse.vstack(pieces, format="csc")


def get_scores(
    expr: np.ndarray,
    pairs: np.ndarray,
    neighbours: List,
    het_vals: np.ndarray,
    min_expr: float,
    spot_indices: npt.NDArray[np.int32],
) -> npt.NDArray[np.float64]:
    """Calculates the scores.
    Parameters
    ----------
    expr: np.ndarray
        Spots*Genes expression of the genes in the pairs.
    pairs: np.ndarray
        Pairs*2 column indices into expr, as (ligand, receptor).
    neighbours: List
        List of np.array's indicating neighbours by indices for each spot.
    het_vals:  np.ndarray
        Spots*Het counts
    min_expr: float
        Minimum expression for gene to be considered expressed.
    spot_indices: npt.NDArray[np.int32]
        Subset of spots to score, given as their integer row positions.
    Returns
    -------
    spot_scores: np.ndarray
        Spots*LR pair of the LR scores per spot.
    """
    return score_spots(
        expr, pairs, adjacency(neighbours), het_vals, min_expr, spot_indices
    )
