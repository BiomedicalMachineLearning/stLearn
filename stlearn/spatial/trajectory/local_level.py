import numpy as np
from anndata import AnnData
from scipy.spatial.distance import cdist

from stlearn.utils import _check_cluster


def local_level(
    adata: AnnData,
    use_label: str = "leiden",
    *,
    cluster: str,
    w: float = 0.5,
    return_matrix: bool = False,
    verbose: bool = True,
) -> np.ndarray | None:
    """\
    Perform local spatial trajectory inference (requires running pseudotime first).

    Parameters
    ----------
    adata:
        Annotated data matrix.
    use_label:
        Use label result of cluster method.
    cluster:
        Label of the cluster in ``adata.obs[use_label]`` to infer the local
        trajectory for, e.g. ``"6"``.
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

    adata.uns["nonabs_dpt_distance_matrix"]: np.ndarray
        Pseudotime distance (difference between values) matrix

    adata.uns["ST_distance_matrix"]: np.ndarray
        STDM
    """
    cluster = _check_cluster(adata, use_label, cluster)
    if verbose:
        print("Start construct trajectory for cluster " + cluster)

    tmp = adata.obs[adata.obs[use_label] == cluster]
    cluster_data = adata[list(tmp.index)]

    list_cluster = cluster_data.obs["sub_cluster_labels"].unique()
    dpt = []
    sd = []
    centroid_dict = cluster_data.uns["centroid_dict"]
    for i in list_cluster:
        if (
            len(adata.obs[adata.obs["sub_cluster_labels"] == i])
            > adata.uns["threshold_spots"]
        ):
            dpt.append(
                cluster_data.obs[cluster_data.obs["sub_cluster_labels"] == i][
                    "dpt_pseudotime"
                ].max()
            )
            sd.append(centroid_dict[i])

    if len(dpt) < 2:
        raise ValueError(
            f"Cluster {cluster!r} has fewer than two subclusters with more than "
            f"{adata.uns['threshold_spots']} spots, so it has no local trajectory. "
            "Choose another cluster, or re-run pseudotime with a smaller eps or "
            "threshold_spots."
        )

    dpt_values = np.asarray(dpt, dtype=float)
    non_abs_dm = dpt_values[:, None] - dpt_values[None, :]
    dm = np.abs(non_abs_dm)
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
