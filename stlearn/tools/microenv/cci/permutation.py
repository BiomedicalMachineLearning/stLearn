import sys, os, random, scipy
import numpy as np
from numba import jit, njit, float64, int64, prange
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from anndata import AnnData
from .base import lr, calc_neighbours, lr_core, get_spot_lrs
from .merge import merge

def permutation(
    adata: AnnData,
    n_pairs: int = 200,
    distance: int = None,
    use_lr: str = "cci_lr",
    use_het: str = None,
    neg_binom: bool = False,
    adj_method: str = 'fdr',
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
    elif type(bg_pairs)==type(None):
        pairs = get_rand_pairs(adata, genes, n_pairs, lrs=adata.uns['lr'])
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

    # for each randomly selected pair, run through cci analysis and keep the scores
    query_pair = adata.uns["lr"]

    # If neighbours not inputted, then compute #
    if type(neighbours) == type(None):
        neighbours = calc_neighbours(adata, distance, index=run_fast)

    if not run_fast and type(background)==type(None): #Run original way if 'fast'=False argument inputted.
        background = []
        for item in pairs:
            adata.uns["lr"] = [item]
            lr(adata, use_lr=use_lr, distance=distance, verbose=False,
                                                neighbours=neighbours, **kwargs)
            if use_het != None:
                merge(adata, use_lr=use_lr, use_het=use_het, verbose=False)
                background += adata.obsm["merged"].tolist()
            else:
                background += adata.obsm[use_lr].tolist()
        background = np.array(background)

    elif type(background)==type(None): #Run fast if background not inputted
        spot_lr1s = get_spot_lrs(adata, pairs, lr_order=True, filter_pairs=False)
        spot_lr2s = get_spot_lrs(adata, pairs, lr_order=False,filter_pairs=False)

        het_vals = np.array([1]*len(adata)) if use_het==None else adata.obsm[use_het]
        background = get_scores(spot_lr1s.values, spot_lr2s.values, neighbours,
                                het_vals).ravel()

    # log back the original query
    adata.uns["lr"] = query_pair

    ##### Negative Binomial fit - dosn't make sense, distribution not neg binom
    pvals, pvals_adj, log10_pvals, lr_sign = get_stats(scores, background,
                                                          neg_binom, adj_method)

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
        adata.obsm["lr_sign"] = lr_sign # scores for spots with pval_adj < 0.05

        # enablePrint()
        print("Results of permutation test has been kept in adata.obsm['lr_pvalues']")
        print("Significant merged result has been kept in adata.obsm['lr_sign']")

    # return adata
    return background

def get_stats(scores, background, neg_binom=False, adj_method='fdr'):
    ##### Negative Binomial fit - dosn't make sense, distribution not neg binom
    if neg_binom:
        pmin, pmax = min(background), max(background)
        background2 = [item - pmin for item in background]
        x = np.linspace(pmin, pmax, 1000)
        res = sm.NegativeBinomial(
            background2, np.ones(len(background2)), loglike_method="nb2"
        ).fit(start_params=[0.1, 0.3])
        mu = res.predict()  # use if not constant
        mu = np.exp(res.params[0])
        alpha = res.params[1]
        Q = 0
        size = 1.0 / alpha * mu ** Q
        prob = size / (size + mu)

        # Calculate probability for all spots
        pvals = 1 - scipy.stats.nbinom.cdf(scores - pmin, size, prob)

    else:  ###### Using the actual values to estimate p-values
        pvals = np.array([len(np.where(background >= score)[0])/len(background)
                                                           for score in scores])

    pvals_adj = multipletests(pvals, method=adj_method)[1]
    log10_pvals_adj = -np.log10(pvals_adj)
    lr_sign = scores * (pvals_adj < 0.05)
    return pvals, pvals_adj, log10_pvals_adj, lr_sign

def get_valid_genes(adata, n_pairs):
    genes = np.array([
        item
        for item in adata.var_names.tolist()
        if not (item.startswith("MT-") or item.startswith("MT_"))
    ])
    if n_pairs >= len(genes) / 2:
        raise ValueError(
            "Too many genes pairs selected, please reduce to a smaller number."
        )
    return genes

def get_rand_pairs(adata: AnnData, genes: np.array, n_pairs: int,
                   lrs: list = None, im: int = None,
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
    lr_genes = [lr.split('_')[0] for lr in lrs]
    lr_genes += [lr.split('_')[1] for lr in lrs]

    # get the position of the median of the means between the two genes
    means_ordered, genes_ordered = get_ordered(adata, genes)
    if type(im) == type(None): #Single background per lr pair mode
        l, r = lrs[0].split('_')
        im = get_median_index(l, r, means_ordered.values, genes_ordered)

    # get n_pair genes sorted by distance to im
    selected = (
        abs(means_ordered - means_ordered[im])
            .sort_values()
            .drop(lr_genes)[: n_pairs * 2]
            .index.tolist()
    )
    selected = selected[0:n_pairs*2]
    adata.uns["selected"] = selected
    # form gene pairs from selected randomly
    random.shuffle(selected)
    pairs = [i + "_" + j for i, j in
             zip(selected[:n_pairs], selected[-n_pairs:])]

    return pairs

def get_ordered(adata, genes):
    means_ordered = adata.to_df()[genes].mean().sort_values()
    genes_ordered = means_ordered.index.values
    return means_ordered, genes_ordered

def get_median_index(l, r, means_ordered, genes_ordered):
    """"Retrieves the index of the gene with a mean expression between the two genes in the lr pair.
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
    i1 = np.where(genes_ordered==l)[0][0]
    i2 = np.where(genes_ordered==r)[0][0]
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

@njit(parallel=True)
def get_scores(
        spot_lr1s: np.ndarray,
        spot_lr2s: np.ndarray,
        neighbours: np.array,
        het_vals: np.array,
) -> np.array:
    """Permutation test for merged result
    Parameters
    ----------
    spot_lr1s: np.ndarray   Spots*GeneOrder1, in format l1, r1, ... ln, rn
    spot_lr2s: np.ndarray   Spots*GeneOrder2, in format r1, l1, ... rn, ln
    het_vals:  np.ndarray   Spots*Het counts
    neighbours: list        List of the neighbours for each spot, if None then computed. Useful for speeding up function.
    Returns
    -------
    spot_scores: np.ndarray   Spots*LR pair of the LR scores per spot.
    """
    spot_scores = np.zeros((spot_lr1s.shape[0], spot_lr1s.shape[1]//2),
                                                                     np.float64)
    for i in prange(0, spot_lr1s.shape[1]//2):
        i_ = i*2 # equivalent to range(0, spot_lr1s.shape[1], 2)
        spot_lr1, spot_lr2 = spot_lr1s[:,i_:(i_+2)], spot_lr2s[:,i_:(i_+2)]
        lr_scores = lr_core(spot_lr1, spot_lr2, neighbours)
        # The merge scores #
        lr_scores = np.multiply(het_vals, lr_scores)
        spot_scores[:, i] = lr_scores
    return spot_scores

# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore printing
def enablePrint():
    sys.stdout = sys.__stdout__
