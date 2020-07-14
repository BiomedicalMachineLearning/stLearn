import sys, os, random, scipy
import numpy as np
import pandas as pd
from anndata import AnnData
from .base import lr
from .merge import merge


# Permutation test for merged result

def permutation(
    adata: AnnData,
    n_pairs: int = 1000,
    distance: int = 30,
    use_data: str = 'normalized',
    threshold: int = 1,
    use_lr: str = 'lr_neighbours_louvain_max',
    use_het: str = 'het'
) -> AnnData:
    """ Merge results from cell type heterogeneity and L-R clustering
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count
    n_pairs: int            Number of gene pairs to run permutation test (default: 1000)
    distance: int           Distance between spots (default: 30)
    use_data: str           Data used for lr clustering (default: 'normalized')
    threshold: int          Threshold in lr clustering (default: 1)
    use_lr: str             LR cluster used for permutation test (default: 'lr_neighbours_louvain_max')
    use_het: str            cell type diversity counts used for permutation test (default 'het')
    Returns
    -------
    adata: AnnData          Data Frame of p-values from permutation test for each window stored in adata.uns['merged_pvalues']
                            Final significant merged scores stored in adata.uns['merged_sign']
    """
    
    blockPrint()

    #  select n_pair*2 closely expressed genes from the data
    genes = [item for item in adata.obsm[use_data].columns.tolist() if not item.startswith('MT_')]
    if n_pairs >= len(genes) / 2:
        raise ValueError('Too many genes pairs selected, please reduce to a smaller number.')
    elif len(adata.uns['lr']) > 1:
        raise ValueError('Permutation test only supported for one LR pair scenario.')
    else:
        means = adata.obsm[use_data][genes].mean().sort_values()
        lr1 = adata.uns['lr'][0].split('_')[0]
        lr2 = adata.uns['lr'][0].split('_')[1]
        i1, i2 = means.index.get_loc(lr1), means.index.get_loc(lr2)
        if means[lr1] > means[lr2]:
            it = i1; i1 = i2; i2 = it
        im = np.where(means.values==means.iloc[i1:i2+1].median())[0][0]
        new1 = means.iloc[im-n_pairs:im].sample(frac=1)
        new2 = means.iloc[im:im+n_pairs].sample(frac=1)
        try:
            new1 = new1.drop(i0)
        except:
            pass
        try:
            new2 = new2.drop(i1)
        except:
            pass
        
        # form gene pairs
        pairs = list(new1.index + '_' + new2.index)
        
        # add LR being tested into the list
        pairs = adata.uns['lr'] + pairs

    original = adata.uns['merged']

    # for each randomly selected pair, run through cci analysis and keep the scores
    scores = []
    for i, item in enumerate(pairs):
        if i > 0:
            adata.uns['lr'] = [item]
            lr(adata, use_data=use_data, threshold=threshold, distance=distance);
            merge(adata, use_lr=use_lr, use_het=use_het);
        else:
            pass
        # z-score
        delta = adata.uns['merged'].subtract(adata.uns['merged'].mean().mean())
        std = (delta ** 2).sum().sum() / (delta.shape[0] * delta.shape[1])
        scores.append(delta.div(std))

    num = adata.uns['merged'].shape[0]
    permutation = pd.DataFrame(0, range(num), range(num))
    for i in range(num):
        for j in range(num):
            distribution = []
            for k in range(1, len(scores)):
                # build the list for merged score of spot [i,j] for each of the randomly selected pairs
                distribution.append(scores[k].iloc[i,j])
            # t-test for result of target and randomly selected pairs on every spot
            ttest = scipy.stats.ttest_1samp(distribution, scores[0].iloc[i,j])
            if ttest.statistic < 0:
                permutation.iloc[i,j] = -np.log10(ttest.pvalue+1e-300) + np.log10(num*num)
            else: permutation.iloc[i,j] = 0

    adata.uns['merged'] = original
    adata.uns['merged_pvalues'] = permutation
    adata.uns['merged_sign'] = adata.uns['merged'].mul((permutation > 2).values)

    enablePrint()
    print("Results of permutation test has been kept in adata.uns['merged_pvalues']")
    print("Significant merged result has been kept in adata.uns['merged_sign']")

    return adata


# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore printing
def enablePrint():
    sys.stdout = sys.__stdout__