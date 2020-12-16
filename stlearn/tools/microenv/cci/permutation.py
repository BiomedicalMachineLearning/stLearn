import sys, os, random, scipy
import numpy as np
import pandas as pd
import statsmodels.api as sm
from anndata import AnnData
from .base import lr
from .merge import merge


def permutation(
    adata: AnnData,
    n_pairs: int = 200,
    distance: int = None,
    use_lr: str = "cci_lr",
    use_het: str = "cci_het",
) -> AnnData:

    """Permutation test for merged result
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count
    n_pairs: int            Number of gene pairs to run permutation test (default: 1000)
    distance: int           Distance between spots (default: 30)
    use_lr: str             LR cluster used for permutation test (default: 'lr_neighbours_louvain_max')
    use_het: str            cell type diversity counts used for permutation test (default 'het')
    Returns
    -------
    adata: AnnData          Data Frame of p-values from permutation test for each window stored in adata.uns['merged_pvalues']
                            Final significant merged scores stored in adata.uns['merged_sign']
    """

    blockPrint()

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
        # sort the mean of each gene expression
        means = adata.to_df()[genes].mean().sort_values()
        lr1 = adata.uns["lr"][0].split("_")[0]
        lr2 = adata.uns["lr"][0].split("_")[1]
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
        pairs = [i + "_" + j for i, j in zip(selected[:n_pairs], selected[-n_pairs:])]

    """
    # generate random pairs
    lr1 = adata.uns['lr'][0].split('_')[0]
    lr2 = adata.uns['lr'][0].split('_')[1]
    genes = [item for item in adata.var_names.tolist() if not (item.startswith('MT-') or item.startswith('MT_') or item==lr1 or item==lr2)]
    random.shuffle(genes)
    pairs = [i + '_' + j for i, j in zip(genes[:n_pairs], genes[-n_pairs:])]
    """

    scores = adata.uns["merged"]
    background = []

    # for each randomly selected pair, run through cci analysis and keep the scores
    for item in pairs:
        adata.uns["lr"] = [item]
        lr(adata, use_lr=use_lr, distance=distance)
        merge(adata, use_lr=use_lr, use_het=use_het)
        background += adata.uns["merged"].tolist()

    # Permutation test for each spot across all runs
    permutation = pd.DataFrame(0, adata.obs_names, ["pval"])

    # Negative Binomial fit
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
    permutation["pval"] = [
        item - np.log10(len(adata.obs_names))
        for item in -np.log10(1 - scipy.stats.nbinom.cdf(scores - pmin, size, prob))
    ]

    adata.uns["merged"] = scores
    adata.uns["merged_pvalues"] = permutation["pval"]
    adata.uns["merged_sign"] = (
        adata.uns["merged"] * (permutation > -np.log10(0.05))["pval"]
    )  # p-value < 0.05

    enablePrint()
    print("Results of permutation test has been kept in adata.uns['merged_pvalues']")
    print("Significant merged result has been kept in adata.uns['merged_sign']")

    return adata


# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore printing
def enablePrint():
    sys.stdout = sys.__stdout__
