from anndata import AnnData
from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from natsort import natsorted


def localization(
    adata: AnnData,
    use_label: str = "louvain",
    eps: int = 20,
    min_samples: int = 0,
    copy: bool = False,
) -> Optional[AnnData]:

    """\
    Perform local clustering by using DBSCAN.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_label
        Use label result of clustering method.
    eps
        The maximum distance between two samples for one to be considered as 
        in the neighborhood of the other. This is not a maximum bound on the 
        distances of points within a cluster. This is the most important DBSCAN 
        parameter to choose appropriately for your data set and distance function.
    min_samples
        The number of samples (or total weight) in a neighborhood for a point to be
        considered as a core point. This includes the point itself.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Anndata
    """

    if "sub_cluster_labels" in adata.obs.columns:
        adata.obs = adata.obs.drop("sub_cluster_labels", axis=1)

    pd.set_option('mode.chained_assignment', None)
    subclusters = pd.Series()
    for i in adata.obs[use_label].unique():

        tmp = adata.obs[adata.obs[use_label] == i]

        clustering = DBSCAN(eps=eps, min_samples=0, algorithm="kd_tree").fit(
            tmp[["imagerow", "imagecol"]])

        labels = clustering.labels_

        sublabels = []
        for label in labels:
            sublabels.append(str(i)+"_"+str(label))
        tmp["sub_labels"] = sublabels
        subclusters = subclusters.append(tmp["sub_labels"])
    pd.reset_option('mode.chained_assignment')

    adata.obs = pd.merge(adata.obs, pd.DataFrame(
        {"sub_cluster_labels": subclusters}),
        left_index=True, right_index=True)
    # Convert to numeric
    converted = dict(enumerate(adata.obs["sub_cluster_labels"].unique()))
    inv_map = {v: k for k, v in converted.items()}
    adata.obs["sub_cluster_labels"] = adata.obs["sub_cluster_labels"].replace(
        inv_map)

    adata.obs["sub_cluster_labels"] = pd.Categorical(
        values=np.array(adata.obs["sub_cluster_labels"]).astype('U'),
        categories=natsorted(
            np.unique(np.array(adata.obs["sub_cluster_labels"])).astype('U')),
    )

    return adata if copy else None
