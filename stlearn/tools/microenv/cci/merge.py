import numpy as np
import pandas as pd
from anndata import AnnData
from .het import create_windows


# Merge the L-R interaction results with cell type heterogeneity results to locate highly interacting sites

def merge(
    adata: AnnData,
    use_lr: str = 'lr_neighbours_louvain_max',
    use_het: str = 'het',
) -> AnnData:
    """ Merge results from cell type heterogeneity and L-R clustering
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count
    use_lr: str             The largest expressed L-R cluster
    use_het: str            The count result of cell type heterogneity
                
    Returns
    -------
    adata: AnnData          With merged result stored in adata.uns['merged']
    """
    coor = adata.obs[['imagerow', 'imagecol']]
    num_row, num_col = adata.uns[use_het].shape[0], adata.uns[use_het].shape[1]
    windows, width, height = create_windows(adata, num_row, num_col)
    counts = pd.DataFrame(0, range(num_row), range(num_col))
    for n, window in enumerate(windows):
        spots = coor[(coor['imagecol'] > window[0]) & (coor['imagecol'] < window[0] + width) \
                    & (coor['imagerow'] < window[1]) & (coor['imagerow'] > window[1] - height)]
        try:
            counts.iloc[num_row-1-n%num_row, n//num_row] = sum(adata.obs[use_lr.split('_max')[0]][spots.index.tolist()] == adata.uns[use_lr]) / len(spots)
        except:
            counts.iloc[num_row-1-n%num_row, n//num_row] = 0
            
    # z-score for counts and adata.uns[use_het]
    delta = counts.subtract(counts.mean().mean())
    std = np.sqrt((delta ** 2).sum().sum() / (delta.shape[0] * delta.shape[1]))
    counts = delta.div(std)

    delta = adata.uns[use_het].subtract(adata.uns[use_het].mean().mean())
    std = np.sqrt((delta ** 2).sum().sum() / (delta.shape[0] * delta.shape[1]))
    
    adata.uns['merged'] = (counts + delta.div(std)).div(2)

    print("Results of spatial interaction analysis has been written to adata.uns['merged']")

    return adata
