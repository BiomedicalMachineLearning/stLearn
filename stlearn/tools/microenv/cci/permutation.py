import sys, os, random, scipy
import numpy as np
from numba import jit, njit, float64, int64
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from anndata import AnnData
from .base import lr, calc_neighbours, lr_core
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
    run_fast: bool = False,
    bg_pairs: list = None,
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
    genes = [
        item
        for item in adata.var_names.tolist()
        if not (item.startswith("MT-") or item.startswith("MT_"))
    ]
    if n_pairs >= len(genes) / 2:
        raise ValueError(
            "Too many genes pairs selected, please reduce to a smaller number."
        )
    elif len(adata.uns["lr"]) > 1:
        raise ValueError("Permutation test only supported for one LR pair scenario.")
    else:
        pairs = get_rand_pairs(adata, adata.uns["lr"], genes, n_pairs)

    pairs = pairs if type(bg_pairs)==type(None) else bg_pairs

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

    if not run_fast: #Run original way if 'fast'=False argument inputted.
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

    else: #Run fast
        df = adata.to_df()
        pairs_rev = [f'{pair.split("_")[1]}_{pair.split("_")[0]}'
                                                              for pair in pairs]
        pairs_wRev = []
        for i in range(len(pairs)):
            pairs_wRev.extend([pairs[i], pairs_rev[i]])

        spot_lr1s = df[[pair.split('_')[1] for pair in pairs_wRev]]
        spot_lr2s = df[[pair.split('_')[0] for pair in pairs_wRev]]
        het_vals = np.array([1]*len(adata)) if use_het==None else adata.obsm[use_het]
        background = get_background(spot_lr1s.values, spot_lr2s.values,
                                                           neighbours, het_vals)

    # log back the original query
    adata.uns["lr"] = query_pair

    # Permutation test for each spot across all runs
    permutation = pd.DataFrame(0, adata.obs_names, ["pval"])

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
        permutation["pval"] = -np.log10(
            multipletests(
                1 - scipy.stats.nbinom.cdf(scores - pmin, size, prob),
                method=adj_method
            )[1]
        )
    else: ###### Using the actual values to estimate p-values
        pvals = np.array([len(np.where(background >= score)[0])/len(background)
                                                           for score in scores])
        permutation["pval"] = -np.log10(multipletests(pvals,
                                                          method=adj_method)[1])
        res = None

    if use_het != None:
        adata.obsm["merged"] = scores
        adata.obsm["merged_pvalues"] = permutation["pval"].values
        adata.obsm["merged_sign"] = (
            adata.obsm["merged"] * (permutation > -np.log10(0.05))["pval"].values
        )  # p-value < 0.05

        # enablePrint()
        print(
            "Results of permutation test has been kept in adata.obsm['merged_pvalues']"
        )
        print("Significant merged result has been kept in adata.obsm['merged_sign']")
    else:
        adata.obsm[use_lr] = scores
        adata.obsm["lr_pvalues"] = permutation["pval"].values
        adata.obsm["lr_sign"] = (
            adata.obsm[use_lr] * (permutation > -np.log10(0.05))["pval"].values
        )  # p-value < 0.05

        # enablePrint()
        print("Results of permutation test has been kept in adata.obsm['lr_pvalues']")
        print("Significant merged result has been kept in adata.obsm['lr_sign']")

    # return adata
    return res, background

def get_rand_pairs(adata, lr, genes, n_pairs):
    # sort the mean of each gene expression
    means = adata.to_df()[genes].mean().sort_values()
    lr1 = lr[0].split("_")[0]
    lr2 = lr[0].split("_")[1]
    i1, i2 = means.index.get_loc(lr1), means.index.get_loc(lr2)
    if means[lr1] > means[lr2]:
        it = i1
        i1 = i2
        i2 = it

    # get the position of the median of the means between the two genes
    im = np.argmin(abs(means.values - means.iloc[i1:i2].median()))

    # get n_pair genes sorted by distance to im
    selected = (
        abs(means - means.iloc[im])
            .sort_values()
            .drop([lr1, lr2])[: n_pairs * 2]
            .index.tolist()
    )
    adata.uns["selected"] = selected
    # form gene pairs from selected randomly
    random.shuffle(selected)
    pairs = [i + "_" + j for i, j in
             zip(selected[:n_pairs], selected[-n_pairs:])]

    return pairs

@njit(parallel=True)
def get_background(
        spot_lr1s: np.ndarray,
        spot_lr2s: np.ndarray,
        neighbours: np.array,
        het_vals: np.array,
) -> np.array:
    """Permutation test for merged result
    Parameters
    ----------
    spot_lr1s: np.ndarray   Spots*Random gene 1
    spot_lr2s: np.ndarray   Spots*Random gene 2
    het_vals:  np.ndarray   Spots*Het counts
    neighbours: list        List of the neighbours for each spot, if None then computed. Useful for speeding up function.
    Returns
    -------
    background: list        The background scores from the random pairs.
    """
    background = np.zeros((1, spot_lr1s.shape[0]*spot_lr1s.shape[1]),
                                                                np.float64)[0,:]
    index = 0
    for i in range(0, spot_lr1s.shape[1], 2):
        spot_lr1, spot_lr2 = spot_lr1s[:,i:(i+2)], spot_lr2s[:,i:(i+2)]
        lr_bg = lr_core(spot_lr1, spot_lr2, neighbours)
        # The merge scores #
        lr_bg = np.multiply(het_vals, lr_bg)
        index_fin = (index+spot_lr1s.shape[0])
        background[index:index_fin] = lr_bg
        index = index_fin
    return background

# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore printing
def enablePrint():
    sys.stdout = sys.__stdout__
