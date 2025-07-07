import numpy as np
import pandas as pd
from anndata import AnnData
from natsort import natsorted
from sklearn.cluster import DBSCAN


def localization(
    adata: AnnData,
    use_label: str = "louvain",
    eps: float = 20.0,
    min_samples: int = 1,
    copy: bool = False,
) -> AnnData | None:
    """\
    Perform local cluster by using DBSCAN.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    use_label: str, default = "louvain"
        Use label result of cluster method.
    eps: float, default 20.0
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. This is not a maximum bound on the
        distances of points within a cluster. This is the most important DBSCAN
        parameter to choose appropriately for your data set and distance function.
    min_samples: int, default = 1
        The number of samples (or total weight) in a neighborhood for a point to be
        considered as a core point. This includes the point itself. Passed into DBSCAN's
        min_samples parameter.
    copy: bool, default = False
        Return a copy instead of writing to adata.
    Returns
    -------
    Anndata
    """

    adata = adata.copy() if copy else adata

    if "sub_cluster_labels" in adata.obs.columns:
        adata.obs = adata.obs.drop("sub_cluster_labels", axis=1)

    pd.set_option("mode.chained_assignment", None)
    subclusters_list = []
    for i in adata.obs[use_label].unique():
        tmp = adata.obs[adata.obs[use_label] == i]

        clustering = DBSCAN(eps=eps, min_samples=min_samples, algorithm="kd_tree").fit(
            tmp[["imagerow", "imagecol"]]
        )

        labels = clustering.labels_

        sublabels = []
        for label in labels:
            sublabels.append(str(i) + "_" + str(label))
        tmp["sub_labels"] = sublabels
        subclusters_list.append(tmp["sub_labels"])

    subclusters = pd.concat(subclusters_list)
    pd.reset_option("mode.chained_assignment")

    adata.obs = pd.merge(
        adata.obs,
        pd.DataFrame({"sub_cluster_labels": subclusters}),
        left_index=True,
        right_index=True,
    )

    # Convert to numeric
    converted = dict(enumerate(adata.obs["sub_cluster_labels"].unique()))
    inv_map = {v: str(k) for k, v in converted.items()}
    adata.obs["sub_cluster_labels"] = adata.obs["sub_cluster_labels"].replace(inv_map)

    adata.obs["sub_cluster_labels"] = pd.Categorical(
        values=np.array(adata.obs["sub_cluster_labels"]).astype("U"),
        categories=natsorted(
            np.unique(np.array(adata.obs["sub_cluster_labels"])).astype("U")
        ),
    )

    labels_cat = list(map(int, adata.obs[use_label].cat.categories))
    cat_ind = {labels_cat[i]: i for i in range(len(labels_cat))}
    adata.uns[use_label + "_index_dict"] = cat_ind

    return adata if copy else None
