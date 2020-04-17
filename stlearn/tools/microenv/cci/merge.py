import numpy as np
import pandas as pd
from anndata import AnnData
from .het import create_windows


# Merge the L-R interaction results with cell type heterogeneity results to locate highly interacting sites

def merge(
    adata: AnnData,
    use_lr: str = 'lr_neighbours_louvain_max',
    use_het: pd.DataFrame,
) -> AnnData:
    """ Merge results from cell type heterogeneity and L-R clustering
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count
    use_lr: str             The largest expressed L-R cluster
    use_het: pd.DataFrame   The counts result of cell type heterogneity
                
    Returns
    -------
    counts: pd.DataFrame    The counts of specified clusters in each window of the tissue
    """
    coor = adata.obs[['imagerow', 'imagecol']]
    num = use_het.shape[0]
    windows, width, height = create_windows(adata, num)
    counts = pd.DataFrame(0, range(num), range(num))
    for n, window in enumerate(windows):
        spots = coor[(coor['imagecol'] > window[0]) & (coor['imagecol'] < window[0] + width) \
                    & (coor['imagerow'] < window[1]) & (coor['imagerow'] > window[1] - height)]
        try:
            counts.iloc[num-1-n%num, n//num] = sum(adata.obs[use_lr.split('_max')[0]][spots.index.tolist()] == adata.uns[use_lr]) / len(spots)
        except:
            counts.iloc[num-1-n%num, n//num] = 0
    # z-score for counts and use_het
    delta = counts.subtract(counts.mean().mean())
    std = (delta ** 2).sum().sum() / (delta.shape[0] * delta.shape[1])
    counts = delta.div(std)

    delta = use_het.subtract(use_het.mean().mean())
    std = (delta ** 2).sum().sum() / (delta.shape[0] * delta.shape[1])
    use_het = delta.div(std)

    adata.uns['merged'] = (counts + use_het).div(2)

    print("Results of spatial interaction analysis has been written to adata.uns['merged']")

    return adata
