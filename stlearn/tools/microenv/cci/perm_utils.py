import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, canberra
from sklearn.preprocessing import MinMaxScaler

from numba import njit, prange
from numba.typed import List

from .base import get_lrs_scores


def nonzero_quantile(expr, q, interpolation):
    """Calculating the non-zero quantiles."""
    nonzero_expr = expr[expr > 0]
    quants = np.quantile(nonzero_expr, q=q, interpolation=interpolation)
    if type(quants) != np.array and type(quants) != np.ndarray:
        quants = np.array([quants])
    return quants


def getzero_prop(expr):
    """Calculating the proportion of zeros."""
    zero_bool = expr == 0
    n_zeros = len(np.where(zero_bool)[0])
    zero_prop = [n_zeros / len(expr)]
    return zero_prop


def get_lr_quants(
    lr_expr: pd.DataFrame,
    l_indices: list,
    r_indices: list,
    quantiles: np.array,
    method="",
):
    """Gets the quantiles per gene in the LR pair, & then concatenates.
    Returns
    -------
    lr_quants, l_quants, r_quants: np.ndarray   First is concatenation of two latter. Each row is a quantile value, each column is a LR pair.
    """

    quant_func = nonzero_quantile if method != "quantiles" else np.quantile

    # First getting the quantiles of gene expression #
    gene_quants = np.apply_along_axis(
        quant_func, 0, lr_expr.values, q=quantiles, interpolation="nearest"
    )

    l_quants = gene_quants[:, l_indices]
    r_quants = gene_quants[:, r_indices]

    lr_quants = np.concatenate((l_quants, r_quants), 0).transpose()

    return lr_quants, l_quants, r_quants


def get_lr_zeroprops(lr_expr: pd.DataFrame, l_indices: list, r_indices: list):
    """Gets the proportion of zeros per gene in the LR pair, & then concatenates.
    Returns
    -------
    lr_props, l_props, r_props: np.ndarray   First is concatenation of two latter. Each row is a prop value, each column is a LR pair.
    """

    # First getting the quantiles of gene expression #
    gene_props = np.apply_along_axis(getzero_prop, 0, lr_expr.values)

    l_props = gene_props[:, l_indices]
    r_props = gene_props[:, r_indices]

    lr_props = np.concatenate((l_props, r_props), 0).transpose()

    return lr_props, l_props, r_props


def get_lr_bounds(lr_value: float, bin_bounds: np.array):
    """For the given lr_value, returns the bin where it belongs.
    Returns
    -------
    lr_bin: tuple   Tuple of length 2, first is the lower bound of the bin, second is upper bound of the bin.
    """
    if np.any(bin_bounds == lr_value):  # If sits on a boundary
        lr_i = np.where(bin_bounds == lr_value)[0][0]
        if lr_value == max(bin_bounds):  # Must be in the final bin
            _lower = bin_bounds[-2]
            _upper = bin_bounds[-1]
        else:  # In the lower bin
            _lower = bin_bounds[lr_i]
            _upper = bin_bounds[lr_i + 1]
    else:  # Bin where it's value sit in-between #
        _lower = bin_bounds[np.where(bin_bounds < lr_value)[0][-1]]
        _upper = bin_bounds[np.where(bin_bounds > lr_value)[0][0]]

    return (_lower, _upper)


def get_similar_genes(
    ref_quants: np.array,
    ref_props: np.array,
    n_genes: int,
    candidate_expr: np.ndarray,
    candidate_genes: np.array,
    quantiles=(0.5),  # (.5, .75, .85, .9, .95, .97, .98, .99, 1)
):
    """Gets genes with a similar expression distribution as the inputted gene,
        by measuring distance between the gene expression quantiles.
    Parameters
    ----------
    ref_quants: np.array     The pre-calculated quantiles.
    ref_props: np.array      The query zero proportions.
    n_genes: int            Number of equivalent genes to select.
    candidate_expr: np.ndarray  Expression of gene candidates (cells*genes).
    candidate_genes: np.array   Same as candidate_expr.shape[1], indicating gene names.
    quantiles: tuple    The quantile to use
    Returns
    -------
    similar_genes: np.array Array of strings for gene names.
    """
    if type(quantiles) == float:
        quantiles = np.array([quantiles])
    else:
        quantiles = np.array(quantiles)

    # Query quants #
    query_quants = np.apply_along_axis(
        nonzero_quantile, 0, candidate_expr, q=quantiles, interpolation="nearest"
    )

    # Need to min-max normalise so can take distance with the proportion #
    all_quants = np.concatenate((np.array([ref_quants]), query_quants), axis=1)
    scaler = MinMaxScaler()
    scaled_quants = scaler.fit_transform(all_quants.transpose()).transpose()
    ref_scaled = scaled_quants[:, 0]
    query_scaled = scaled_quants[:, 1:]

    # Query props #
    query_props = np.apply_along_axis(getzero_prop, 0, candidate_expr)

    # Concatenating to create the ref & query vals to match #
    ref_vals = np.array([ref_scaled[0], ref_props[0]])  # both between 0 & 1
    query_vals = np.concatenate((query_scaled, query_props))

    # Measuring distances from the desired gene #
    dists = np.apply_along_axis(canberra, 0, query_vals, ref_vals)

    # Retrieving desired number of genes #
    order = np.argsort(dists)
    similar_genes = candidate_genes[order[0:n_genes]]

    """ During debugging, plotting distribution of distances & selected genes.
    import matplotlib.pyplot as plt
    cutoff = dists[order[n_genes]]
    plt.hist(dists, bins=100)
    plt.vlines(cutoff, 0, 500, color='r')
    plt.show()
    """

    return similar_genes


def get_similar_genes_Quantiles(
    gene_expr: np.array,
    n_genes: int,
    candidate_quants: np.ndarray,
    candidate_genes: np.array,
    quantiles=(0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 1),
):
    """Gets genes with a similar expression distribution as the inputted gene,
        by measuring distance between the gene expression quantiles.
    Parameters
    ----------
    gene_expr: np.array     Expression of the gene of interest, or, if the same length as quantiles, then assumes is the pre-calculated quantiles.
    n_genes: int            Number of equivalent genes to select.
    candidate_quants: np.ndarray  Expression quantiles of gene candidates (quantiles*genes).
    candidate_genes: np.array   Same as candidate_expr.shape[1], indicating gene names.
    quantiles: tuple    The quantile to use
    Returns
    -------
    similar_genes: np.array Array of strings for gene names.
    """

    if type(quantiles) == float:
        quantiles = np.array([quantiles])
    else:
        quantiles = np.array(quantiles)

    # Getting the quantiles for the gene #
    if len(gene_expr) != len(quantiles):
        # ref_quants = np.quantile(gene_expr, q=quantiles, interpolation='nearest')
        ref_quants = nonzero_quantile(gene_expr, q=quantiles, interpolation="nearest")
    else:
        ref_quants = gene_expr

    # Measuring distances from the desired gene #
    dists = np.apply_along_axis(canberra, 0, candidate_quants, ref_quants)
    order = np.argsort(dists)

    """ During debugging, plotting distribution of distances & selected genes.
    import matplotlib.pyplot as plt
    cutoff = dists[order[n_genes]]
    fig, ax = plt.subplots()
    ax.hist(dists[order[0:28000]], bins=1000)
    y_max = ax.get_ylim()[1]
    ax.vlines(cutoff, 0, y_max/2, color='r')
    plt.show()
    print(candidate_quants[:,order[0:3]]) # Showing the quantiles of selected
    print(candidate_quants[:,order[n_genes-3:n_genes]])
    print(ref_quants)
    """

    # Retrieving desired number of genes #
    similar_genes = candidate_genes[order[0:n_genes]]

    return similar_genes


@njit(parallel=True)
def get_similar_genesFAST(
    ref_quants: np.array,
    n_genes: int,
    candidate_quants: np.ndarray,
    candidate_genes: np.array,
):
    """Fast version of the above with parallelisation."""

    # Measuring distances from the desired gene #
    dists = np.zeros((1, candidate_quants.shape[1]), dtype=np.float64)[0, :]
    for i in prange(0, candidate_quants.shape[1]):
        cand_quants = candidate_quants[:, i]
        abs_diff = ref_quants - cand_quants
        abs_diff[abs_diff < 0] = -abs_diff[abs_diff < 0]
        dists[i] = np.nansum(abs_diff / (ref_quants + cand_quants))

    # Need to remove the zero-dists since this indicates they are expressed
    # exactly the same, & hence likely in the same spot !!!
    nonzero_bool = dists > 0
    dists = dists[nonzero_bool]
    candidate_quants = candidate_quants[:, nonzero_bool]
    candidate_genes = candidate_genes[nonzero_bool]
    order = np.argsort(dists)

    """ During debugging, plotting distribution of distances & selected genes.
    import matplotlib.pyplot as plt
    cutoff = dists[order[n_genes]]
    fig, ax = plt.subplots()
    ax.hist(dists[order[0:28000]], bins=1000)
    y_max = ax.get_ylim()[1]
    ax.vlines(cutoff, 0, y_max/2, color='r')
    plt.show()
    print(candidate_quants[:,order[0:3]]) # Showing the quantiles of selected
    print(candidate_quants[:,order[n_genes-3:n_genes]])
    print(ref_quants)
    """

    # Retrieving desired number of genes #
    similar_genes = candidate_genes[order[0:n_genes]]

    return similar_genes


@njit
def gen_rand_pairs(genes1: np.array, genes2: np.array, n_pairs: int):
    """Generates random pairs of genes."""

    rand_pairs = List()
    for j in range(0, n_pairs):
        l_rand = np.random.choice(genes1, 1)[0]
        r_rand = np.random.choice(genes2, 1)[0]
        rand_pair = "_".join([l_rand, r_rand])
        while rand_pair in rand_pairs or l_rand == r_rand:
            l_rand = np.random.choice(genes1, 1)[0]
            r_rand = np.random.choice(genes2, 1)[0]
            rand_pair = "_".join([l_rand, r_rand])

        rand_pairs.append(rand_pair)

    return rand_pairs


def get_lr_features(adata, lr_expr, lrs, quantiles):
    """Gets expression features of LR pairs; nonzero-median, zero-prop, quantiles."""
    quantiles = np.array(quantiles)

    # Determining indices of LR pairs #
    l_indices, r_indices = [], []
    for lr in lrs:
        l_, r_ = lr.split("_")
        l_indices.extend(np.where(lr_expr.columns.values == l_)[0])
        r_indices.extend(np.where(lr_expr.columns.values == r_)[0])

    # The nonzero median when quantiles=.5 #
    lr_quants, l_quants, r_quants = get_lr_quants(
        lr_expr, l_indices, r_indices, quantiles, method="quantiles"
    )

    # Calculating the zero proportions, for grouping based on median/zeros #
    lr_props, l_props, r_props = get_lr_zeroprops(lr_expr, l_indices, r_indices)

    ######## Getting lr features for later diagnostics #######
    lr_meds, l_meds, r_meds = get_lr_quants(
        lr_expr, l_indices, r_indices, quantiles=np.array([0.5]), method=""
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
    lr_features = pd.DataFrame(index=lrs, columns=cols)
    lr_features.iloc[:, 0] = lr_median_means
    lr_features.iloc[:, 1] = lr_prop_means
    lr_features.iloc[:, 2] = np.array(median_ranks)
    lr_features.iloc[:, 3] = np.array(prop_ranks)
    lr_features.iloc[:, 4] = np.array(mean_ranks)
    lr_features = lr_features.iloc[np.argsort(mean_ranks), :]
    lr_cols = [f"L_{quant}" for quant in quantiles] + [
        f"R_{quant}" for quant in quantiles
    ]
    quant_df = pd.DataFrame(lr_quants, columns=lr_cols, index=lrs)
    lr_features = pd.concat((lr_features, quant_df), axis=1)
    adata.uns["lrfeatures"] = lr_features

    return lr_features


def get_lr_bg(
    adata,
    neighbours,
    het_vals,
    min_expr,
    lr_,
    lr_score,
    l_quant,
    r_quant,
    genes,
    candidate_quants,
    gene_bg_genes,
    n_genes,
    n_pairs,
):
    """Gets the LR-specific background & bg spot indices."""
    l_, r_ = lr_.split("_")
    if l_ not in gene_bg_genes:
        l_genes = get_similar_genesFAST(
            l_quant, n_genes, candidate_quants, genes  # group_l_props,
        )
        gene_bg_genes[l_] = l_genes
    else:
        l_genes = gene_bg_genes[l_]

    if r_ not in gene_bg_genes:
        r_genes = get_similar_genesFAST(
            r_quant, n_genes, candidate_quants, genes  # group_r_props,
        )
        gene_bg_genes[r_] = r_genes
    else:
        r_genes = gene_bg_genes[r_]

    rand_pairs = gen_rand_pairs(l_genes, r_genes, n_pairs)
    spot_indices = np.where(lr_score > 0)[0]

    background = get_lrs_scores(
        adata,
        rand_pairs,
        neighbours,
        het_vals,
        min_expr,
        filter_pairs=False,
        spot_indices=spot_indices,
    )

    return background, spot_indices
