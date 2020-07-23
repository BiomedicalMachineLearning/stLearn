from anndata import AnnData
from typing import Optional, Union
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist
import scanpy

def pseudotime(
    adata: AnnData,
    use_label: str = "louvain",
    eps: float = 20,
    threshold: float = 0.01,
    radius: int = 50,
    method: str = "mean",
    threshold_spots: int = 5,
    use_sme: bool = False,
    copy: bool = False,
) -> Optional[AnnData]:

    """\
    Perform pseudotime analysis.

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
    threshold
        Threshold to find the significant connection for PAGA graph.
    radius
        radius to adjust data for diffusion map
    method
        method to adjust the data.
    use_sme
        Use adjusted feature by SME normalization or not
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Anndata
    """

    try:
        del adata.obsm["X_diffmap"]
    except:
        pass
    try:
        del adata.obsm["X_draw_graph_fr"]
    except:
        pass

    # Localize
    from stlearn.spatials.clustering import localization
    localization(adata, use_label=use_label, eps=eps)

    # Running paga
    scanpy.tl.paga(adata, groups=use_label)

    # Denoising the graph
    scanpy.tl.diffmap(adata)

    if use_sme:
        from stlearn.spatials.morphology import adjust
        adjust(adata, use_data="X_diffmap", radius=radius, method=method)
        adata.obsm["X_diffmap"] = adata.obsm["X_diffmap_morphology"]


    # Get connection matrix
    cnt_matrix = adata.uns["paga"]["connectivities"].toarray()

    # Filter by threshold
    threshold = threshold

    cnt_matrix[cnt_matrix < threshold] = 0.
    cnt_matrix = pd.DataFrame(cnt_matrix)

    # Mapping louvain label to subcluster
    split_node = {}
    for label in adata.obs[use_label].unique():
        meaningful_sub = []
        for i in adata.obs[adata.obs[use_label] == label]["sub_cluster_labels"].unique():
            if len(adata.obs[adata.obs["sub_cluster_labels"] == str(i)]) > threshold_spots:
                meaningful_sub.append(i)

        split_node[int(label)] = meaningful_sub
            
    adata.uns["threshold_spots"] = threshold_spots
    adata.uns["split_node"] = split_node
    # Replicate louvain label row to prepare for subcluster connection 
    # matrix construction
    replicate_list = np.array([])
    for i in range(0, len(cnt_matrix)):
        replicate_list = np.concatenate(
            [replicate_list, np.array([i]*len(split_node[i]))])

    # Connection matrix for subcluster
    cnt_matrix = cnt_matrix.loc[replicate_list.astype(
        int), replicate_list.astype(int)]

    # Replace column and index
    cnt_matrix.columns = replace_with_dict(cnt_matrix.columns, split_node)
    cnt_matrix.index = replace_with_dict(cnt_matrix.index, split_node)

    # Sort column and index
    cnt_matrix = cnt_matrix.loc[selection_sort(np.array(cnt_matrix.columns)),
                                selection_sort(np.array(cnt_matrix.index))]

    # Create a connection graph of subclusters
    G = nx.from_pandas_adjacency(cnt_matrix)

    adata.uns['global_graph'] = G

    # Create centroid dict for subclusters
    from sklearn.neighbors import NearestCentroid
    clf = NearestCentroid()
    clf.fit(adata.obs[["imagecol", "imagerow"]].values,
            adata.obs["sub_cluster_labels"])
    centroid_dict = dict(zip(clf.classes_.astype(int), clf.centroids_))

    def closest_node(node, nodes):
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - node)**2, axis=1)
        return np.argmin(dist_2)

    for cl in adata.obs["sub_cluster_labels"].unique():
        cl_points = adata.obs[adata.obs["sub_cluster_labels"] == cl][["imagecol","imagerow"]].values
        new_centroid = cl_points[closest_node(centroid_dict[int(cl)],cl_points)]
        centroid_dict[int(cl)] = new_centroid

    adata.uns["centroid_dict"] = centroid_dict

    # Running diffusion pseudo-time
    scanpy.tl.dpt(adata)

    return adata if copy else None


######## utils ########

def replace_with_dict(ar, dic):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    out = np.zeros_like(ar)
    for key, val in zip(k, v):
        out[ar == key] = val
    return out


def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x
