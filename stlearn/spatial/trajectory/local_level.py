import numpy as np
from anndata import AnnData
from scipy.spatial.distance import cdist


def local_level(
    adata: AnnData,
    use_label: str = "louvain",
    cluster: int = 9,
    w: float = 0.5,
    return_matrix: bool = False,
    verbose: bool = True,
) -> np.ndarray | None:
    """\
    Perform local sptial trajectory inference (required run pseudotime first).

    Parameters
    ----------
    adata:
        Annotated data matrix.
    use_label:
        Use label result of cluster method.
    cluster:
        Choose cluster to perform local spatial trajectory inference.
    w: float, default=0.5
        Pseudo-spatio-temporal distance weight (balance between spatial effect and DPT)
    return_matrix:
        Return PTS matrix for local level
    verbose : bool, default=True
        Whether to print progress information.
    Returns
    -------
    np.ndarray: the STDM (spatio-temporal distance matrix) - weighted combination of
    spatial and temporal distances.

    adata["nonabs_dpt_distance_matrix"]: np.ndarray
        Pseudotime distance (difference between values) matrix

    adata["nonabs_dpt_distance_matrix"]: np.ndarray
        STDM
    """
    if verbose:
        print("Start construct trajectory for subcluster " + str(cluster))

    tmp = adata.obs[adata.obs[use_label] == str(cluster)]
    cluster_data = adata[list(tmp.index)]

    list_cluster = cluster_data.obs["sub_cluster_labels"].unique()
    dpt = []
    sd = []
    centroid_dict = cluster_data.uns["centroid_dict"]
    centroid_dict = {int(key): centroid_dict[key] for key in centroid_dict}
    for i in list_cluster:
        if (
            len(adata.obs[adata.obs["sub_cluster_labels"] == str(i)])
            > adata.uns["threshold_spots"]
        ):
            dpt.append(
                cluster_data.obs[cluster_data.obs["sub_cluster_labels"] == i][
                    "dpt_pseudotime"
                ].max()
            )
            sd.append(centroid_dict[int(i)])
    dm = cdist(
        np.array(dpt).reshape(-1, 1),
        np.array(dpt).reshape(-1, 1),
        lambda u, v: np.abs(u - v),
    )

    non_abs_dm = cdist(
        np.array(dpt).reshape(-1, 1), np.array(dpt).reshape(-1, 1), lambda u, v: u - v
    )
    adata.uns["nonabs_dpt_distance_matrix"] = non_abs_dm

    scale_dm = dm / np.max(dm)
    sdm = cdist(np.array(sd), np.array(sd), "euclidean")
    scale_sdm = sdm / np.max(sdm)

    stdm = scale_dm * w + scale_sdm * (1 - w)
    adata.uns["ST_distance_matrix"] = stdm

    if return_matrix:
        return stdm
    else:
        return None
