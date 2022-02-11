import sys, os, random, scipy
import numpy as np
import pandas as pd
from numba.typed import List
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering

from anndata import AnnData
from .base import lr, calc_neighbours, get_spot_lrs, get_lrs_scores, get_scores
from .merge import merge
from .perm_utils import get_lr_features, get_lr_bg

# Newest method #
def perform_spot_testing(
    adata: AnnData,
    lr_scores: np.ndarray,
    lrs: np.array,
    n_pairs: int,
    neighbours: List,
    het_vals: np.array,
    min_expr: float,
    adj_method: str = "fdr_bh",
    pval_adj_cutoff: float = 0.05,
    verbose: bool = True,
    save_bg=False,
    neg_binom=False,
    quantiles=(0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.9975, 0.999, 1),
):
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
            "Exiting since need atleast "
            f"{n_genes} genes to generate {n_pairs} pairs."
        )
        return

    if n_pairs < 100:
        print(
            "Exiting since n_pairs<100, need much larger number of pairs to "
            "get accurate backgrounds (e.g. 1000)."
        )
        return

    ####### Quantiles to select similar gene to LRs to gen. rand-pairs #######
    lr_expr = adata[:, lr_genes].to_df()
    lr_feats = get_lr_features(adata, lr_expr, lrs, quantiles)
    l_quants = lr_feats.loc[
        lrs, [col for col in lr_feats.columns if "L_" in col]
    ].values
    r_quants = lr_feats.loc[
        lrs, [col for col in lr_feats.columns if "R_" in col]
    ].values
    candidate_quants = np.apply_along_axis(
        np.quantile, 0, candidate_expr, q=quantiles, interpolation="nearest"
    )
    # Ensuring consistent typing to prevent numba errors #
    l_quants = l_quants.astype("<f4")
    r_quants = r_quants.astype("<f4")
    candidate_quants = candidate_quants.astype("<f4")

    ######## Background per LR, but only for spots where LR has a score ########
    # Determine the indices of the spots where each LR has a score #
    cols = ["n_spots", "n_spots_sig", "n_spots_sig_pval"]
    lr_summary = np.zeros((lr_scores.shape[1], 3), np.int)
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
        disable=verbose == False,
    ) as pbar:

        gene_bg_genes = {}  # Keep track of genes which can be used to gen. rand-pairs.
        spot_lr_indices = [
            [] for i in range(lr_scores.shape[0])
        ]  # tracks the lrs tested in a given spot for MHT !!!!
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
            )

            if save_bg:
                adata.uns["lrs_to_bg"][lr_] = background
                adata.uns["lr_spot_indices"][lr_] = spot_indices

            if not neg_binom:  # Calculate empirical p-values per-spot
                for spot_i, spot_index in enumerate(spot_indices):
                    n_greater = len(
                        np.where(background[spot_i, :] >= lr_scores[spot_index, lr_j])[
                            0
                        ]
                    )
                    n_greater = n_greater if n_greater != 0 else 1  # pseudocount
                    pvals[spot_index, lr_j] = n_greater / background.shape[1]
                    spot_lr_indices[spot_index].append(lr_j)
            else:  # Fitting NB per LR
                lr_j_scores = lr_scores[spot_indices, lr_j]
                bg_ = background.ravel()
                bg_wScore = np.array(list(lr_j_scores) + list(bg_))

                ##### 1) rounding discretisation
                # First multiple to get minimum value to be one before rounding #
                bg_1 = bg_wScore * (1 / min(bg_wScore[bg_wScore != 0]))
                bg_1 = np.round(bg_1)
                lr_j_scores_1 = bg_1[0 : len(lr_j_scores)]
                bg_1 = bg_1[len(lr_j_scores) : len(bg_1)]

                ###### Getting the pvalue from negative binomial approach
                round_pvals, _, _, _ = get_stats(
                    lr_j_scores_1,
                    bg_1,
                    len(bg_1),
                    neg_binom=True,
                    return_negbinom_params=False,
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
                    pvals[spot_i, lr_indices], method=adj_method
                )[1]

            log10pvals_adj[spot_i, :] = -np.log10(pvals_adj[spot_i, :])

            # Recording per lr results for this LR #
            lrs_in_spot = lr_scores[spot_i] > min_expr
            sig_lrs_in_spot = pvals_adj[spot_i, :] < pval_adj_cutoff
            sigpval_lrs_in_spot = pvals[spot_i, :] < pval_adj_cutoff
            lr_summary[lrs_in_spot, 0] += 1
            lr_summary[sig_lrs_in_spot, 1] += 1
            lr_summary[sigpval_lrs_in_spot, 2] += 1

            lr_sig_scores[spot_i, sig_lrs_in_spot == False] = 0

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
            "rows in adata.uns['lr_summary']."
        )
        print("Summary of LR results in adata.uns['lr_summary'].")


# Version 2, no longer in use, see above for newest method #
def perform_perm_testing(
    adata: AnnData,
    lr_scores: np.ndarray,
    n_pairs: int,
    lrs: np.array,
    lr_mid_dist: int,
    verbose: float,
    neighbours: List,
    het_vals: np.array,
    min_expr: float,
    neg_binom: bool,
    adj_method: str,
    pval_adj_cutoff: float,
):
    """Performs the grouped permutation testing when taking the stats approach."""
    if n_pairs != 0:  # Perform permutation testing
        # Grouping spots with similar mean expression point #
        genes = get_valid_genes(adata, n_pairs)
        means_ordered, genes_ordered = get_ordered(adata, genes)
        ims = np.array(
            [
                get_median_index(
                    lr_.split("_")[0],
                    lr_.split("_")[1],
                    means_ordered.values,
                    genes_ordered,
                )
                for lr_ in lrs
            ]
        ).reshape(-1, 1)

        if len(lrs) > 1:  # Multi-LR pair mode, group LRs to generate backgrounds
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=lr_mid_dist,
                affinity="manhattan",
                linkage="single",
            )
            lr_groups = clusterer.fit_predict(ims)
            lr_group_set = np.unique(lr_groups)
            if verbose:
                print(f"{len(lr_group_set)} lr groups with similar expression levels.")

        else:  # Single LR pair mode, generate background for the LR.
            lr_groups = np.array([0])
            lr_group_set = lr_groups

        res_info = ["lr_scores", "p_val", "p_adj", "-log10(p_adj)", "lr_sig_scores"]
        n_, n_sigs = np.array([0] * len(lrs)), np.array([0] * len(lrs))
        per_lr_results = {}
        with tqdm(
            total=len(lr_group_set),
            desc="Generating background distributions for the LR pair groups..",
            bar_format="{l_bar}{bar} [ time left: {remaining} ]",
        ) as pbar:
            for group in lr_group_set:
                # Determining common mid-point for each group #
                group_bool = lr_groups == group
                group_im = int(np.median(ims[group_bool, 0]))

                # Calculating the background #
                rand_pairs = get_rand_pairs(adata, genes, n_pairs, lrs=lrs, im=group_im)
                background = get_lrs_scores(
                    adata,
                    rand_pairs,
                    neighbours,
                    het_vals,
                    min_expr,
                    filter_pairs=False,
                ).ravel()
                total_bg = len(background)
                background = background[background != 0]  # Filtering for increase speed

                # Getting stats for each lr in group #
                group_lr_indices = np.where(group_bool)[0]
                for lr_i in group_lr_indices:
                    lr_ = lrs[lr_i]
                    lr_results = pd.DataFrame(index=adata.obs_names, columns=res_info)
                    scores = lr_scores[:, lr_i]
                    stats = get_stats(
                        scores,
                        background,
                        total_bg,
                        neg_binom,
                        adj_method,
                        pval_adj_cutoff=pval_adj_cutoff,
                    )
                    full_stats = [scores] + list(stats)
                    for vals, colname in zip(full_stats, res_info):
                        lr_results[colname] = vals

                    n_[lr_i] = len(np.where(scores > 0)[0])
                    n_sigs[lr_i] = len(
                        np.where(lr_results["p_adj"].values < pval_adj_cutoff)[0]
                    )
                    if n_sigs[lr_i] > 0:
                        per_lr_results[lr_] = lr_results
                pbar.update(1)

        print(f"{len(per_lr_results)} LR pairs with significant interactions.")

        lr_summary = pd.DataFrame(index=lrs, columns=["n_spots", "n_spots_sig"])
        lr_summary["n_spots"] = n_
        lr_summary["n_spots_sig"] = n_sigs
        lr_summary = lr_summary.iloc[np.argsort(-n_sigs)]

    else:  # Simply store the scores
        per_lr_results = {}
        lr_summary = pd.DataFrame(index=lrs, columns=["n_spots"])
        for i, lr_ in enumerate(lrs):
            lr_results = pd.DataFrame(index=adata.obs_names, columns=["lr_scores"])
            lr_results["lr_scores"] = lr_scores[:, i]
            per_lr_results[lr_] = lr_results
            lr_summary.loc[lr_, "n_spots"] = len(np.where(lr_scores[:, i] > 0)[0])
        lr_summary = lr_summary.iloc[np.argsort(-lr_summary.values[:, 0]), :]

    adata.uns["lr_summary"] = lr_summary
    adata.uns["per_lr_results"] = per_lr_results
    if verbose:
        print(
            "Summary of significant spots for each lr pair in adata.uns['lr_summary']."
        )
        print(
            "Spot enrichment statistics of LR interactions in adata.uns['per_lr_results']"
        )


# No longer in use #
def permutation(
    adata: AnnData,
    n_pairs: int = 200,
    distance: int = None,
    use_lr: str = "cci_lr",
    use_het: str = None,
    neg_binom: bool = False,
    adj_method: str = "fdr",
    neighbours: list = None,
    run_fast: bool = True,
    bg_pairs: list = None,
    background: np.array = None,
    **kwargs,
) -> AnnData:

    """Permutation test for merged result
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count
    n_pairs: int            Number of gene pairs to run permutation test (default: 1000)
    distance: int           Distance between spots (default: 30)
    use_lr: str             LR cluster used for permutation test (default: 'lr_neighbours_louvain_max')
    use_het: str            cell type diversity counts used for permutation test (default 'het')
    neg_binom: bool         Whether to fit neg binomial paramaters to bg distribution for p-val est.
    adj_method: str         Method used by statsmodels.stats.multitest.multipletests for MHT correction.
    neighbours: list        List of the neighbours for each spot, if None then computed. Useful for speeding up function.
    **kwargs:               Extra arguments parsed to lr.
    Returns
    -------
    adata: AnnData          Data Frame of p-values from permutation test for each window stored in adata.uns['merged_pvalues']
                            Final significant merged scores stored in adata.uns['merged_sign']
    """

    # blockPrint()

    #  select n_pair*2 closely expressed genes from the data
    genes = get_valid_genes(adata, n_pairs)
    if len(adata.uns["lr"]) > 1:
        raise ValueError("Permutation test only supported for one LR pair scenario.")
    elif type(bg_pairs) == type(None):
        pairs = get_rand_pairs(adata, genes, n_pairs, lrs=adata.uns["lr"])
    else:
        pairs = bg_pairs

    """
    # generate random pairs
    lr1 = adata.uns['lr'][0].split('_')[0]
    lr2 = adata.uns['lr'][0].split('_')[1]
    genes = [item for item in adata.var_names.tolist() if not (item.startswith('MT-') or item.startswith('MT_') or item==lr1 or item==lr2)]
    random.shuffle(genes)
    pairs = [i + '_' + j for i, j in zip(genes[:n_pairs], genes[-n_pairs:])]
    """
    if use_het != None:
        scores = adata.obsm["merged"]
    else:
        scores = adata.obsm[use_lr]

    # for each randomly selected pair, run through cci_rank analysis and keep the scores
    query_pair = adata.uns["lr"]

    # If neighbours not inputted, then compute #
    if type(neighbours) == type(None):
        neighbours = calc_neighbours(adata, distance, index=run_fast)

    if not run_fast and type(background) == type(
        None
    ):  # Run original way if 'fast'=False argument inputted.
        background = []
        for item in pairs:
            adata.uns["lr"] = [item]
            lr(
                adata,
                use_lr=use_lr,
                distance=distance,
                verbose=False,
                neighbours=neighbours,
                **kwargs,
            )
            if use_het != None:
                merge(adata, use_lr=use_lr, use_het=use_het, verbose=False)
                background += adata.obsm["merged"].tolist()
            else:
                background += adata.obsm[use_lr].tolist()
        background = np.array(background)

    elif type(background) == type(None):  # Run fast if background not inputted
        spot_lr1s = get_spot_lrs(adata, pairs, lr_order=True, filter_pairs=False)
        spot_lr2s = get_spot_lrs(adata, pairs, lr_order=False, filter_pairs=False)

        het_vals = (
            np.array([1] * len(adata)) if use_het == None else adata.obsm[use_het]
        )
        background = get_scores(
            spot_lr1s.values, spot_lr2s.values, neighbours, het_vals
        ).ravel()

    # log back the original query
    adata.uns["lr"] = query_pair

    #### Negative Binomial fit
    pvals, pvals_adj, log10_pvals, lr_sign = get_stats(
        scores, background, neg_binom, adj_method
    )

    if use_het != None:
        adata.obsm["merged"] = scores
        adata.obsm["merged_pvalues"] = log10_pvals
        adata.obsm["merged_sign"] = lr_sign

        # enablePrint()
        print(
            "Results of permutation test has been kept in adata.obsm['merged_pvalues']"
        )
        print("Significant merged result has been kept in adata.obsm['merged_sign']")
    else:
        adata.obsm[use_lr] = scores
        adata.obsm["lr_pvalues"] = log10_pvals
        adata.obsm["lr_sign"] = lr_sign  # scores for spots with pval_adj < 0.05

        # enablePrint()
        print("Results of permutation test has been kept in adata.obsm['lr_pvalues']")
        print("Significant merged result has been kept in adata.obsm['lr_sign']")

    # return adata
    return background


def get_stats(
    scores: np.array,
    background: np.array,
    total_bg: int,
    neg_binom: bool = False,
    adj_method: str = "fdr_bh",
    pval_adj_cutoff: float = 0.01,
    return_negbinom_params: bool = False,
):
    """Retrieves valid candidate genes to be used for random gene pairs.
    Parameters
    ----------
    scores: np.array        Per spot scores for a particular LR pair.
    background: np.array    Background distribution for non-zero scores.
    total_bg: int           Total number of background values calculated.
    neg_binom: bool         Whether to use neg-binomial distribution to estimate p-values, NOT appropriate with log1p data, alternative is to use background distribution itself (recommend higher number of n_pairs for this).
    adj_method: str         Parsed to statsmodels.stats.multitest.multipletests for multiple hypothesis testing correction.
    Returns
    -------
    stats: tuple          Per spot pvalues, pvals_adj, log10_pvals_adj, lr_sign (the LR scores for significant spots).
    """
    ##### Negative Binomial fit
    if neg_binom:
        # Need to make full background for fitting !!!
        background = np.array(list(background) + [0] * (total_bg - len(background)))
        pmin, pmax = min(background), max(background)
        background2 = [item - pmin for item in background]
        x = np.linspace(pmin, pmax, 1000)
        res = sm.NegativeBinomial(
            background2, np.ones(len(background2)), loglike_method="nb2"
        ).fit(start_params=[0.1, 0.3], disp=0)
        mu = res.predict()  # use if not constant
        mu = np.exp(res.params[0])
        alpha = res.params[1]
        Q = 0
        size = 1.0 / alpha * mu**Q
        prob = size / (size + mu)

        if return_negbinom_params:  # For testing purposes #
            return size, prob

        # Calculate probability for all spots
        pvals = 1 - scipy.stats.nbinom.cdf(scores - pmin, size, prob)

    else:  ###### Using the actual values to estimate p-values
        pvals = np.zeros((1, len(scores)), dtype=np.float)[0, :]
        nonzero_score_bool = scores > 0
        nonzero_score_indices = np.where(nonzero_score_bool)[0]
        zero_score_indices = np.where(nonzero_score_bool == False)[0]
        pvals[zero_score_indices] = (total_bg - len(background)) / total_bg
        pvals[nonzero_score_indices] = [
            len(np.where(background >= scores[i])[0]) / total_bg
            for i in nonzero_score_indices
        ]

    pvals_adj = multipletests(pvals, method=adj_method)[1]
    log10_pvals_adj = -np.log10(pvals_adj)
    lr_sign = scores * (pvals_adj < pval_adj_cutoff)
    return pvals, pvals_adj, log10_pvals_adj, lr_sign


# @njit(parallel=True)
# def perm_pvals(scores, background):
#     """Determines the p-values based on the actual observed frequency of the \
#         indicated score or greater in the background.
#     """
#     pvals = np.zeros((1, len(scores)), np.float64)[0,:]
#     for i in prange(len(pvals)):


def get_valid_genes(adata: AnnData, n_pairs: int) -> np.array:
    """Retrieves valid candidate genes to be used for random gene pairs.
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count
    n_pairs: int            The number of random pairs will generate elsewhere.
    Returns
    -------
    genes: np.array          List of genes which could be valid pairs.
    """
    genes = np.array(
        [
            item
            for item in adata.var_names.tolist()
            if not (item.startswith("MT-") or item.startswith("MT_"))
        ]
    )
    if n_pairs >= len(genes) / 2:
        raise ValueError(
            "Too many genes pairs selected, please reduce to a smaller number."
        )
    return genes


def get_rand_pairs(
    adata: AnnData,
    genes: np.array,
    n_pairs: int,
    lrs: list = None,
    im: int = None,
):
    """Gets equivalent random gene pairs for the inputted lr pair.
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count
    lr: int            The lr pair string to get equivalent random pairs for (e.g. 'L_R')
    genes: np.array           Candidate genes to use as pairs.
    n_pairs: int             Number of random pairs to generate.
    Returns
    -------
    pairs: list          List of random gene pairs with equivalent mean expression (e.g. ['L_R'])
    """
    lr_genes = [lr.split("_")[0] for lr in lrs]
    lr_genes += [lr.split("_")[1] for lr in lrs]

    # get the position of the median of the means between the two genes
    means_ordered, genes_ordered = get_ordered(adata, genes)
    if type(im) == type(None):  # Single background per lr pair mode
        l, r = lrs[0].split("_")
        im = get_median_index(l, r, means_ordered.values, genes_ordered)

    # get n_pair genes sorted by distance to im
    selected = (
        abs(means_ordered - means_ordered[im])
        .sort_values()
        .drop(lr_genes)[: n_pairs * 2]
        .index.tolist()
    )
    selected = selected[0 : n_pairs * 2]
    adata.uns["selected"] = selected
    # form gene pairs from selected randomly
    random.shuffle(selected)
    pairs = [i + "_" + j for i, j in zip(selected[:n_pairs], selected[-n_pairs:])]

    return pairs


def get_ordered(adata, genes):
    means_ordered = adata.to_df()[genes].mean().sort_values()
    genes_ordered = means_ordered.index.values
    return means_ordered, genes_ordered


def get_median_index(l, r, means_ordered, genes_ordered):
    """ "Retrieves the index of the gene with a mean expression between the two genes in the lr pair.
    Parameters
    ----------
    X: np.ndarray          Spot*Gene expression.
    l: str                 Ligand gene.
    r: str                 Receptor gene.
    genes: np.array        Candidate genes to use as pairs.
    Returns
    -------
    pairs: list          List of random gene pairs with equivalent mean expression (e.g. ['L_R'])
    """
    # sort the mean of each gene expression
    i1 = np.where(genes_ordered == l)[0][0]
    i2 = np.where(genes_ordered == r)[0][0]
    if means_ordered[i1] > means_ordered[i2]:
        it = i1
        i1 = i2
        i2 = it

    im = np.argmin(np.abs(means_ordered - np.median(means_ordered[i1:i2])))
    return im
    # means = adata.to_df()[genes].mean().sort_values()
    # lr1 = lr[0].split("_")[0]
    # lr2 = lr[0].split("_")[1]
    # i1, i2 = means.index.get_loc(lr1), means.index.get_loc(lr2)
    # if means[lr1] > means[lr2]:
    #     it = i1
    #     i1 = i2
    #     i2 = it
    #
    # # get the position of the median of the means between the two genes
    # im = np.argmin(abs(means.values - means.iloc[i1:i2]))


# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore printing
def enablePrint():
    sys.stdout = sys.__stdout__
