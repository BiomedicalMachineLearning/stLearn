import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import statsmodels.api as sm
from anndata import AnnData
from numba.typed import List
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from .perm_utils import get_lr_bg, get_lr_features


def perform_spot_testing(
    adata: AnnData,
    lr_scores: np.ndarray,
    lrs: npt.NDArray[np.str_],
    n_pairs: int,
    neighbours: List,
    het_vals: np.ndarray,
    min_expr: float,
    adj_method: str = "fdr_bh",
    pval_adj_cutoff: float = 0.05,
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
) -> None:
    """Calls significant spots by creating random gene pairs with similar
    expression to given LR pair; only generate background for spots
    which have score for given LR.
    """
    quantiles = np.array(quantiles)

    lr_genes = np.unique([lr_.split("_") for lr_ in lrs])
    genes = np.array([gene for gene in adata.var_names if gene not in lr_genes])
    candidate_expr = adata[:, genes].to_df().values

    n_genes = round(np.sqrt(n_pairs) * 2)
    if len(genes) < n_genes:
        print(
            f"Exiting since need atleast {n_genes} genes to generate {n_pairs} pairs.",
        )
        return

    if n_pairs < 100:
        print(
            "Exiting since n_pairs<100, need much larger number of pairs to "
            "get accurate backgrounds (e.g. 1000).",
        )
        return

    # Quantiles to select similar gene to LRs to gen. rand-pairs
    lr_expr = adata[:, lr_genes].to_df()
    lr_feats = get_lr_features(adata, lr_expr, lrs, quantiles)
    l_quants = lr_feats.loc[
        lrs, [col for col in lr_feats.columns if "L_" in col]
    ].values
    r_quants = lr_feats.loc[
        lrs, [col for col in lr_feats.columns if "R_" in col]
    ].values
    candidate_quants = np.apply_along_axis(
        np.quantile,
        0,
        candidate_expr,
        q=quantiles,
        method="nearest",
    )
    # Ensuring consistent typing to prevent numba errors #
    l_quants = l_quants.astype("<f4")
    r_quants = r_quants.astype("<f4")
    candidate_quants = candidate_quants.astype("<f4")

    # Background per LR, but only for spots where LR has a score
    # Determine the indices of the spots where each LR has a score #
    cols = ["n_spots", "n_spots_sig", "n_spots_sig_pval"]
    lr_summary = np.zeros((lr_scores.shape[1], 3), np.int32)
    pvals = np.ones(lr_scores.shape, dtype=np.float64)
    pvals_adj = np.ones(lr_scores.shape, dtype=np.float64)
    log10pvals_adj = np.zeros(lr_scores.shape, dtype=np.float64)
    lr_sig_scores = lr_scores.copy()

    # If we are saving the backgrounds #
    if save_bg:
        adata.uns["lrs_to_bg"] = {}
        adata.uns["lr_spot_indices"] = {}

    with tqdm(
        total=lr_scores.shape[1],
        desc="Generating backgrounds & testing each LR pair...",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
        disable=not verbose,
    ) as pbar:
        # Keep track of genes which can be used to gen. rand-pairs.
        gene_bg_genes: dict[str, np.ndarray] = {}
        # tracks the lrs tested in a given spot for MHT !!!!
        spot_lr_indices: list[list[int]] = [[] for i in range(lr_scores.shape[0])]
        for lr_j in range(lr_scores.shape[1]):
            lr_ = lrs[lr_j]
            background, spot_indices = get_lr_bg(
                adata,
                neighbours,
                het_vals,
                min_expr,
                lr_,
                lr_scores[:, lr_j],
                l_quants[lr_j, :],
                r_quants[lr_j, :],
                genes,
                candidate_quants,
                gene_bg_genes,
                n_genes,
                n_pairs,
                random_state + lr_j,
            )

            if save_bg:
                adata.uns["lrs_to_bg"][lr_] = background
                adata.uns["lr_spot_indices"][lr_] = spot_indices

            if not neg_binom:  # Calculate empirical p-values per-spot
                for spot_i, spot_index in enumerate(spot_indices):
                    n_greater = len(
                        np.where(background[spot_i, :] >= lr_scores[spot_index, lr_j])[
                            0
                        ],
                    )
                    n_greater = n_greater if n_greater != 0 else 1  # pseudocount
                    pvals[spot_index, lr_j] = n_greater / background.shape[1]
                    spot_lr_indices[spot_index].append(lr_j)
            else:  # Fitting NB per LR
                lr_j_scores = lr_scores[spot_indices, lr_j]
                bg_ = background.ravel()
                bg_wScore = np.array(list(lr_j_scores) + list(bg_))

                # 1) rounding discretisation
                # First multiple to get minimum value to be one before rounding #
                bg_1 = bg_wScore * (1 / min(bg_wScore[bg_wScore != 0]))
                bg_1 = np.round(bg_1)
                lr_j_scores_1 = bg_1[0 : len(lr_j_scores)]
                bg_1 = bg_1[len(lr_j_scores) : len(bg_1)]

                # Getting the pvalue from negative binomial approach
                round_pvals, _, _, _ = get_stats(
                    lr_j_scores_1,
                    bg_1,
                    len(bg_1),
                    neg_binom=True,
                )
                pvals[spot_indices, lr_j] = round_pvals
                for spot_index in spot_indices:
                    spot_lr_indices[spot_index].append(lr_j)

            pbar.update(1)

        # MHT correction # filling in other stats #
        for spot_i in range(lr_scores.shape[0]):
            lr_indices = spot_lr_indices[spot_i]
            if len(lr_indices) != 0:
                pvals_adj[spot_i, lr_indices] = multipletests(
                    pvals[spot_i, lr_indices],
                    method=adj_method,
                )[1]

            log10pvals_adj[spot_i, :] = -np.log10(pvals_adj[spot_i, :])

            # Recording per lr results for this LR #
            lrs_in_spot = lr_scores[spot_i] > min_expr
            sig_lrs_in_spot = pvals_adj[spot_i, :] < pval_adj_cutoff
            sigpval_lrs_in_spot = pvals[spot_i, :] < pval_adj_cutoff
            lr_summary[lrs_in_spot, 0] += 1
            lr_summary[sig_lrs_in_spot, 1] += 1
            lr_summary[sigpval_lrs_in_spot, 2] += 1

            lr_sig_scores[spot_i, ~sig_lrs_in_spot] = 0

    # Ordering the results according to number of significant spots per LR#
    order = np.argsort(-lr_summary[:, 1])
    lrs_ordered = lrs[order]
    lr_summary = lr_summary[order, :]
    lr_summary = pd.DataFrame(lr_summary, index=lrs_ordered, columns=cols)
    lr_scores = lr_scores[:, order]
    pvals = pvals[:, order]
    pvals_adj = pvals_adj[:, order]
    log10pvals_adj = log10pvals_adj[:, order]
    lr_sig_scores = lr_sig_scores[:, order]

    # Saving the results in AnnData #
    if verbose:
        print("\nStoring results:\n")

    adata.uns["lr_summary"] = lr_summary
    res_info = ["lr_scores", "p_vals", "p_adjs", "-log10(p_adjs)", "lr_sig_scores"]
    mats = [lr_scores, pvals, pvals_adj, log10pvals_adj, lr_sig_scores]
    for i, info_name in enumerate(res_info):
        adata.obsm[info_name] = mats[i]
        if verbose:
            print(f"{info_name} stored in adata.obsm['{info_name}'].")

    if verbose:
        print(
            "\nPer-spot results in adata.obsm have columns in same order as "
            "rows in adata.uns['lr_summary'].",
        )
        print("Summary of LR results in adata.uns['lr_summary'].")


def get_stats(
    scores: np.ndarray,
    background: np.ndarray,
    total_bg: int,
    neg_binom: bool = False,
    adj_method: str = "fdr_bh",
    pval_adj_cutoff: float = 0.01,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Retrieves valid candidate genes to be used for random gene pairs.
    Parameters
    ----------
    scores: np.ndarray        Per spot scores for a particular LR pair.
    background: np.ndarray    Background distribution for non-zero scores.
    total_bg: int           Total number of background values calculated.
    neg_binom: bool         Whether to use neg-binomial distribution to estimate
                            p-values, NOT appropriate with log1p data, alternative is
                            to use background distribution itself (recommend higher
                            number of n_pairs for this).
    adj_method: str         Parsed to statsmodels.stats.multitest.multipletests for
                            multiple hypothesis testing correction.
    pval_adj_cutoff: float  Significance cutoff.

    Returns
    -------
    stats: tuple            Per spot pvalues, pvals_adj, log10_pvals_adj, lr_sign
                            (the LR scores for significant spots).
    """
    # Negative Binomial fit
    if neg_binom:
        # Need to make full background for fitting !!!
        background = np.array(list(background) + [0] * (total_bg - len(background)))
        pmin = min(background)
        background2 = [item - pmin for item in background]
        res = sm.NegativeBinomial(
            background2,
            np.ones(len(background2)),
            loglike_method="nb2",
        ).fit(start_params=[0.1, 0.3], disp=0)
        mu = res.predict()  # use if not constant
        mu = np.exp(res.params[0])
        alpha = res.params[1]
        Q = 0
        size = 1.0 / alpha * mu**Q
        prob = size / (size + mu)
        # Calculate probability for all spots
        pvals = 1 - scipy.stats.nbinom.cdf(scores - pmin, size, prob)

    else:
        # Using the actual values to estimate p-values
        pvals = np.zeros((1, len(scores)), dtype=np.float64)[0, :]
        nonzero_score_bool = scores > 0
        nonzero_score_indices = np.where(nonzero_score_bool)[0]
        zero_score_indices = np.where(~nonzero_score_bool)[0]
        pvals[zero_score_indices] = (total_bg - len(background)) / total_bg
        pvals[nonzero_score_indices] = [
            len(np.where(background >= scores[i])[0]) / total_bg
            for i in nonzero_score_indices
        ]

    pvals_adj = multipletests(pvals, method=adj_method)[1]
    log10_pvals_adj = -np.log10(pvals_adj)
    lr_sign = scores * (pvals_adj < pval_adj_cutoff)
    return pvals, pvals_adj, log10_pvals_adj, lr_sign
