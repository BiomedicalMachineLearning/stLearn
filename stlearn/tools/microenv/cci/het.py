import numpy as np
import pandas as pd
from anndata import AnnData

def create_windows(
    adata: AnnData,
    num_row: int,   
    num_col: int,
):
    """ Generate screening windows across the tissue sample
    Parameters
    ----------
    adata: AnnData          The data object to generate windows on
    num_row: int            Number of rows
    num_col: int            Number of columns
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
    width = (max_x - min_x) / num_col
    height = (max_y - min_y) / num_row
    windows = []
    # generate windows from top to bottom and left to right
    for i in range(num_row * num_col):
        x = min_x + i // num_row * width  # left side
        y = max_y - i % num_row * height  # upper side
        windows.append([x, y])

    return windows, width, height


def count(
    adata: AnnData,
    num_row: int = 10,
    num_col: int = 10,
    use_clustering: str = 'louvain',
    use_clusters: list = [],
    use_het: str = 'het'

) -> AnnData:
    """ Count the cell type densities
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count
    num_row: int            Number of windows on height
    num_col: int            Number of windows on width
    use_clustering:         The cell type results to use in counting  
    use_clusters:           Specify certain cluster(s) in counting
    use_het:                The stoarge place for result
    Returns
    -------
    adata: AnnData          With the counts of specified clusters in each window of the tissue stored as adata.uns['het']
    """
    if len(use_clusters) == 0:
        use_clusters = list(set(adata.obs[use_clustering]))
    coor = adata.obs[['imagerow', 'imagecol']]
    cluster = adata.obs[use_clustering]
    windows, width, height = create_windows(adata, num_row, num_col)
    counts = pd.DataFrame(0, range(num_row), range(num_col))
    for n, window in enumerate(windows):
        spots = coor[(coor['imagecol'] > window[0]) & (coor['imagecol'] < window[0] + width) \
                    & (coor['imagerow'] < window[1]) & (coor['imagerow'] > window[1] - height)]
        counts.iloc[num_row-1-n%num_row, n//num_row] = len(set([i for i in cluster[spots.index.tolist()] if i in use_clusters]))

    adata.uns[use_het] = counts

    print("Counts for cluster (cell type) diversity stored into data.uns['het']")
    
    return adata
