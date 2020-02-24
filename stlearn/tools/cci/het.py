import numpy as np
import pandas as pd
from anndata import AnnData

def create_windows(
    adata: AnnData,
    use_data: str = 'filtered_counts',
    num: int = 10
):
    
    data = adata.obsm[use_data]
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
    use_data: str = 'filtered_counts',
    use_cluster: str = 'louvain'
) -> pd.DataFrame:
    data = adata.obsm[use_data]
    coor = adata.obs[['imagerow', 'imagecol']]
    cluster = adata.obs[use_cluster]
    windows, width, height = create_windows(adata, use_data, num)
    counts = pd.DataFrame(0, range(num), range(num))
    for n, window in enumerate(windows):
        spots = coor[(coor['imagecol'] > window[0]) & (coor['imagecol'] < window[0] + width) \
                    & (coor['imagerow'] < window[1]) & (coor['imagerow'] > window[1] - height)]
        counts.iloc[n%num, n//num] = len(set(cluster[spots.index.tolist()]))

    return counts
