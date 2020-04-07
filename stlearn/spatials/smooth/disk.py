from typing import Optional, Union
import numpy as np
from anndata import AnnData
import logging as logg
import scipy.spatial as spatial


def disk(
    adata: AnnData,
    use_data: str = "X_umap",
    radius: float = 10.0,
    rates: int = 1,
    method: str = "mean",
    copy: bool = False,
) -> Optional[AnnData]:

    coor = adata.obs[["imagecol", "imagerow"]]
    count_embed = adata.obsm[use_data]
    point_tree = spatial.cKDTree(coor)

    lag_coor = []
    tmp = []

    for i in range(len(coor)):
        current_neightbor = point_tree.query_ball_point(
            coor.values[i], radius)  # Spatial weight
        # print(coor.values[i])
        tmp.append(current_neightbor)
        # print(coor.values[current_neightbor])
        main = count_embed[current_neightbor]
        current_neightbor.remove(i)
        addition = count_embed[current_neightbor]

        for i in range(0, rates):
            main = np.append(main, addition, axis=0)
        if method == "mean":
            # New umap based on SW
            lag_coor.append(list(np.mean(main, axis=0)))
        elif method == "median":
            # New umap based on SW
            lag_coor.append(list(np.median(main, axis=0)))
        else:
            raise ValueError("Only 'median' and 'mean' are aceptable")

    new_embed = use_data + "_disk"

    adata.obsm[new_embed] = np.array(lag_coor)

    print(
        'Disk smoothing function is applied! The new data are stored in adata.obsm["X_diffmap_disk"]')

    return adata if copy else None
