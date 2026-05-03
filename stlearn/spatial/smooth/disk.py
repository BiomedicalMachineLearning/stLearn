import numpy as np
import scipy.spatial as spatial
from anndata import AnnData


def disk(
    adata: AnnData,
    use_data: str = "X_umap",
    radius: float = 10.0,
    rates: int = 1,
    method: str = "mean",
    copy: bool = False,
) -> AnnData | None:
    adata = adata.copy() if copy else adata

    coords = adata.obs[["imagecol", "imagerow"]]
    count_embed = adata.obsm[use_data]
    point_tree = spatial.cKDTree(coords)

    # Preallocate result
    n_points = len(coords)
    n_features = count_embed.shape[1]
    lag_coords = np.empty((n_points, n_features), dtype=count_embed.dtype)

    # Validate method first
    if method == "mean":
        reducer = np.mean
    elif method == "median":
        reducer = np.median
    else:
        raise ValueError("Only 'median' and 'mean' are acceptable")

    for i in range(len(coords)):
        # Spatial weight
        current_neighbor = point_tree.query_ball_point(coords.values[i], radius)
        main = count_embed[current_neighbor]
        current_neighbor.remove(i)
        addition = count_embed[current_neighbor]

        # Replicate `addition` `rates` times and append in one allocation.
        # Faster than append.
        repeated = np.tile(addition, (rates, 1))
        main = np.concatenate([main, repeated], axis=0)

        # Smoothed feature vector: column-wise mean/median over self + neighbours.
        lag_coords[i] = reducer(main, axis=0)

    new_embed = use_data + "_disk"

    adata.obsm[new_embed] = lag_coords

    print(
        f"Disk smoothing function is applied! The new data are stored in "
        f'adata.obsm["{new_embed}"]'
    )

    return adata if copy else None
