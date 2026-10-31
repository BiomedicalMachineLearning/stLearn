"""Per-spot p-values for each LR pair, against a background of random gene pairs
with similar expression."""

from dataclasses import dataclass
from functools import partial

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import scipy.sparse
import statsmodels.api as sm
from anndata import AnnData
from numba.typed import List
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from ._results import Matrix, scored_entries
from .perm_utils import _column_quantiles, background_pairs, get_lr_features
from .scoring import LRScorer, spots_per_chunk


@dataclass
class SpotTests:
    pvals: scipy.sparse.csc_matrix
    """Spots by LR pairs: a p-value for each spot with a score for the LR, the
    entries of the scores given; 1, left out, for the rest."""
    lr_features: pd.DataFrame
    """Expression features per LR pair, for diagnostics."""
    backgrounds: dict[str, np.ndarray] | None = None
    """Per LR pair, background scores of the scored spots; only with save_bg."""
    bg_spot_indices: dict[str, np.ndarray] | None = None
    """Per LR pair, the spots the background rows belong to; only with save_bg."""


def perform_spot_testing(
    adata: AnnData,
    lr_scores: Matrix,
    lrs: npt.NDArray[np.str_],
    n_pairs: int,
    neighbours: List,
    het_vals: np.ndarray,
    min_expr: float,
    verbose: bool = True,
    save_bg: bool = False,
    neg_binom: bool = False,
    quantiles: tuple[float, ...] = (
        0.5,
        0.75,
        0.85,
        0.9,
        0.95,
        0.97,
        0.98,
        0.99,
        0.995,
        0.9975,
        0.999,
        1,
    ),
    random_state: int = 0,
) -> SpotTests:
    """Tests each spot's LR scores against random gene pairs with similar
    expression to the LR pair; backgrounds are only generated for spots
    which have a score for the LR: those above 0 in lr_scores, sparse (as
    get_lrs_scores returns with sparse_result) or dense. The p-values come back
    sparse, an entry for each.
    """
    quantiles = np.array(quantiles)
    scores = scored_entries(lr_scores)

    lr_genes = np.unique([lr_.split("_") for lr_ in lrs])
    candidates = np.flatnonzero(~adata.var_names.isin(lr_genes))
    genes = adata.var_names.to_numpy()[candidates]
    n_genes = round(np.sqrt(n_pairs) * 2)
    if len(genes) < n_genes:
        raise ValueError(
            f"Need at least {n_genes} non-LR genes to generate {n_pairs} "
            f"random pairs; found {len(genes)}."
        )
    if n_pairs < 100:
        raise ValueError(
            f"n_pairs must be at least 100 (1,000 or more recommended); got {n_pairs}."
        )

    scorer = LRScorer(adata, neighbours)  # its by-column expression serves below too

    # Quantiles to select similar gene to LRs to gen. rand-pairs
    lr_feats = get_lr_features(adata[:, lr_genes], lrs, quantiles)
    l_quants = lr_feats.loc[
        lrs, [col for col in lr_feats.columns if "L_" in col]
    ].values
    r_quants = lr_feats.loc[
        lrs, [col for col in lr_feats.columns if "R_" in col]
    ].values
    candidate_quants = gene_quantiles(
        adata, genes, quantiles, expression=scorer.columns
    )
    # Ensuring consistent typing to prevent numba errors #
    l_quants = l_quants.astype("<f4")
    r_quants = r_quants.astype("<f4")
    candidate_quants = candidate_quants.astype("<f4")

    # Background per LR, but only for spots where LR has a score #
    pvals = scores.copy()
    pvals.data[:] = 1.0
    backgrounds: dict[str, np.ndarray] | None = {} if save_bg else None
    bg_spot_indices: dict[str, np.ndarray] | None = {} if save_bg else None

    with tqdm(
        total=scores.shape[1],
        desc="Generating backgrounds & testing each LR pair...",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
        disable=not verbose,
    ) as pbar:
        # Per LR gene, the candidates (positions) its random pairs draw from.
        gene_bg_genes: dict[str, np.ndarray] = {}
        for lr_j in range(scores.shape[1]):
            lr_ = lrs[lr_j]
            start, end = scores.indptr[lr_j], scores.indptr[lr_j + 1]
            if start == end:
                # Nothing to test: no spot has a score for the LR.
                if backgrounds is not None and bg_spot_indices is not None:
                    backgrounds[lr_] = np.empty((0, n_pairs))
                    bg_spot_indices[lr_] = np.empty(0, dtype=np.int64)
                pbar.update(1)
                continue
            lr_j_scores = scores.data[start:end]
            spot_indices = scores.indices[start:end].astype(np.int64)
            rand_pairs = background_pairs(
                lr_,
                l_quants[lr_j, :],
                r_quants[lr_j, :],
                candidates,
                candidate_quants,
                gene_bg_genes,
                n_genes,
                n_pairs,
                random_state + lr_j,
            )
            score = partial(score_background, scorer, rand_pairs, het_vals, min_expr)

            if backgrounds is not None and bg_spot_indices is not None:
                backgrounds[lr_] = score(spot_indices)
                bg_spot_indices[lr_] = spot_indices
            # Empirical p-values, a chunk of spots at a time
            if not neg_binom:
                chunk = spots_per_chunk(2 * n_genes, n_pairs)
                for a in range(0, len(spot_indices), chunk):
                    background = score(spot_indices[a : a + chunk])
                    n_greater = (background >= lr_j_scores[a : a + chunk, None]).sum(1)
                    n_greater = np.maximum(n_greater, 1)  # pseudocount
                    pvals.data[start + a : start + a + len(background)] = (
                        n_greater / n_pairs
                    )
                    # Collect memory before the next chunk is made
                    del background
            # Fitting NB per LR
            else:
                background = score(spot_indices)
                bg_ = background.ravel()
                bg_w_score = np.array(list(lr_j_scores) + list(bg_))

                # 1) rounding discretisation
                # First multiple to get minimum value to be one before rounding #
                scale = 1.0 / min(bg_w_score[bg_w_score != 0])
                scaled = np.round(bg_w_score * scale)
                n_obs = len(lr_j_scores)
                lr_j_scores_1 = scaled[:n_obs]
                bg_1 = scaled[n_obs:]

                # Getting the pvalue from negative binomial approach
                round_pvals, _, _, _ = get_stats(
                    lr_j_scores_1,
                    bg_1,
                    len(bg_1),
                )
                pvals.data[start:end] = round_pvals
            pbar.update(1)

    return SpotTests(pvals, lr_feats, backgrounds, bg_spot_indices)


def score_background(
    scorer: LRScorer,
    pairs: npt.NDArray[np.int64],
    het_vals: np.ndarray,
    min_expr: float,
    spots: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """An LR's background at these spots: spots by its random pairs' scores."""
    return scorer.score_indices(pairs, het_vals, min_expr, spots)


def gene_quantiles(
    adata: AnnData,
    genes: npt.NDArray[np.str_],
    quantiles: npt.NDArray[np.float64],
    chunk_size: int = 1000,
    expression: scipy.sparse.csc_matrix | None = None,
) -> np.ndarray:
    """Quantiles (rows) of each gene's expression (columns) across spots, as
    np.quantile(..., method="nearest") gives them, `chunk_size` genes at a time.
    Sparse expression is done from its stored values. `expression` is adata.X by
    column (CSC) when the caller has it already: every gene's quantiles are then
    computed and `genes`' taken, rather than copying their columns out first.
    """
    if expression is None:
        expr = adata[:, genes].X
        if scipy.sparse.issparse(expr):
            expr = scipy.sparse.csc_matrix(expr)
        columns = np.arange(len(genes))
    else:
        expr = expression
        columns = adata.var_names.get_indexer(genes)
    by_chunk = [
        _column_quantiles(expr[:, start : start + chunk_size], quantiles, nonzero=False)
        for start in range(0, expr.shape[1], chunk_size)
    ]
    return np.hstack(by_chunk)[:, columns]


def get_stats(
    scores: np.ndarray,
    background: np.ndarray,
    total_bg: int,
    adj_method: str = "fdr_bh",
    pval_adj_cutoff: float = 0.01,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """P-values of the scores under a negative binomial fitted to the background.
    Parameters
    ----------
    scores: np.ndarray        Per spot scores for a particular LR pair.
    background: np.ndarray    Background distribution for non-zero scores.
    total_bg: int           Total number of background values calculated.
    adj_method: str         Parsed to statsmodels.stats.multitest.multipletests for
                            multiple hypothesis testing correction.
    pval_adj_cutoff: float  Significance cutoff.

    Returns
    -------
    stats: tuple            Per spot pvalues, pvals_adj, log10_pvals_adj, lr_sign
                            (the LR scores for significant spots).
    """
    # Negative Binomial fit
    # Need to make full background for fitting !!!
    background = np.array(list(background) + [0] * (total_bg - len(background)))
    pmin = min(background)
    background2 = [item - pmin for item in background]
    res = sm.NegativeBinomial(
        background2,
        np.ones(len(background2)),
        loglike_method="nb2",
    ).fit(start_params=[0.1, 0.3], disp=0)
    mu = np.exp(res.params[0])
    alpha = res.params[1]
    Q = 0
    size = 1.0 / alpha * mu**Q
    prob = size / (size + mu)
    # Calculate probability for all spots
    pvals = 1 - scipy.stats.nbinom.cdf(scores - pmin, size, prob)
    pvals_adj = multipletests(pvals, method=adj_method)[1]
    log10_pvals_adj = -np.log10(pvals_adj)
    lr_sign = scores * (pvals_adj < pval_adj_cutoff)
    return pvals, pvals_adj, log10_pvals_adj, lr_sign
