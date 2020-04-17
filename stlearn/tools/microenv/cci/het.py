import numpy as np
import pandas as pd
from anndata import AnnData

def create_windows(
    adata: AnnData,
    num: int,   
):
    """ Generate screening windows across the tissue sample
    Parameters
    ----------
    adata: AnnData          The data object to generate windows on
    num: int                Number of windows on X- and Y- axes
    Returns
    -------
    windows                 The individual windows defined by left and upper side
    width                   Width of windows
    height                  Height of windows
    """   
    coor = adata.obs[['imagerow', 'imagecol']]
    max_x = max(coor['imagecol'])
    min_x = min(coor['imagecol'])
    max_y = max(coor['imagerow'])
    min_y = min(coor['imagerow'])
    width = (max_x - min_x) / num
    height = (max_y - min_y) / num
    windows = []
    # prepare heatmap data (top down and left to right)
    for i in range(num * num):
        x = min_x + i // num * width  # left side
        y = max_y - i % num * height  # upper side
        windows.append([x, y])

    return windows, width, height


def count(
    adata: AnnData,
    num: int = 10,
    use_clustering: str = 'louvain',
    use_clusters: list = [],
) -> pd.DataFrame:
    """ Count the cell type densities
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count
    num: int                Number of windows on X- and Y-axes
    use_clustering:         The cell type results to use in counting  
    use_clusters:           Specify certain cluster(s) in counting
     
    Returns
    -------
    counts: pd.DataFrame    The counts of specified clusters in each window of the tissue
    """
    if len(use_clusters) == 0:
        use_clusters = list(set(adata.obs[use_clustering]))
    coor = adata.obs[['imagerow', 'imagecol']]
    cluster = adata.obs[use_clustering]
    windows, width, height = create_windows(adata, num)
    counts = pd.DataFrame(0, range(num), range(num))
    for n, window in enumerate(windows):
        spots = coor[(coor['imagecol'] > window[0]) & (coor['imagecol'] < window[0] + width) \
                    & (coor['imagerow'] < window[1]) & (coor['imagerow'] > window[1] - height)]
        counts.iloc[num-1-n%num, n//num] = len(set([i for i in cluster[spots.index.tolist()] if i in use_clusters]))

    return counts
