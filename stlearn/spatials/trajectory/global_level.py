from anndata import AnnData
from typing import Optional, Union
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist
from stlearn.utils import _read_graph
from sklearn.metrics import pairwise_distances


def global_level(
    adata: AnnData,
    use_label: str = "louvain",
    use_rep: str = "X_pca",
    n_dims: int = 40,
    list_clusters: list = [],
    return_graph: bool = False,
    w: float = None,
    verbose: bool = True,
    copy: bool = False,
) -> Optional[AnnData]:

    """\
    Perform global sptial trajectory inference.

    Parameters
    ----------
    adata
        Annotated data matrix.
    list_clusters
        Setup a list of cluster to perform pseudo-space-time
    use_label
        Use label result of cluster method.
    return_graph
        Return PTS graph
    w
        Pseudo-spatio-temporal distance weight (balance between spatial effect and DPT)
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Anndata
    """

    assert w <= 1, "w should be in range 0 to 1"
    # Get global graph
    G = _read_graph(adata, "global_graph")
    # Convert to directed graph
    H = G.to_directed()

    cat_inds = adata.uns[use_label + "_index_dict"]
    inds_cat = {v: k for (k, v) in cat_inds.items()}

    # Query cluster
    if type(list_clusters[0]) == str:
        list_clusters = [cat_inds[label] for label in list_clusters]
    query_nodes = list_clusters

    query_nodes = ordering_nodes(query_nodes, use_label, adata)
    if verbose:
        print(
            "Start to construct the trajectory: "
            + " -> ".join(np.array(query_nodes).astype(str))
        )

    query_dict = {}
    order_dict = {}

    for i in query_nodes:
        order = 0
        for j in adata.obs[adata.obs[use_label] == str(inds_cat[i])][
            "sub_cluster_labels"
        ].unique():
            query_dict[int(j)] = int(i)
            order_dict[int(j)] = int(order)

            order += 1
    dm_list = []
    sdm_list = []
    order_big_dict = {}
    edge_list = []

    for i, j in enumerate(query_nodes):
        order_big_dict[j] = int(i)
        if i == len(query_nodes) - 1:
            break
        for j in adata.uns["split_node"][query_nodes[i]]:
            for k in adata.uns["split_node"][query_nodes[i + 1]]:
                edge_list.append((int(j), int(k)))

        # Calculate DPT distance matrix
        dm_list.append(
            ge_distance_matrix(
                adata,
                inds_cat[query_nodes[i]],
                inds_cat[query_nodes[i + 1]],
                use_label=use_label,
                use_rep=use_rep,
                n_dims=n_dims,
            )
        )
        # Calculate Spatial distance matrix
        sdm_list.append(
            spatial_distance_matrix(
                adata,
                inds_cat[query_nodes[i]],
                inds_cat[query_nodes[i + 1]],
                use_label=use_label,
            )
        )

    # Get centroid dictionary
    centroid_dict = adata.uns["centroid_dict"]
    centroid_dict = {int(key): centroid_dict[key] for key in centroid_dict}

    H_sub = H.edge_subgraph(edge_list)
    if not nx.is_connected(H_sub.to_undirected()):
        raise ValueError(
            "The chosen clusters are not available to construct the spatial trajectory! Please choose other path."
        )
    H_sub = nx.DiGraph(H_sub)
    prepare_root = []
    for node in adata.uns["split_node"][query_nodes[0]]:
        H_sub.add_edge(9999, int(node))
        prepare_root.append(centroid_dict[int(node)])

    prepare_root = np.array(prepare_root)
    centroide = (
        sum(prepare_root[:, 0]) / len(prepare_root[:, 0]),
        sum(prepare_root[:, 1]) / len(prepare_root[:, 1]),
    )

    # Get centroid dictionary
    centroid_dict = adata.uns["centroid_dict"]
    centroid_dict = {int(key): centroid_dict[key] for key in centroid_dict}

    H_sub = H.edge_subgraph(edge_list)
    H_sub = nx.DiGraph(H_sub)

    prepare_root = []
    for node in adata.uns["split_node"][query_nodes[0]]:
        H_sub.add_edge(9999, int(node))
        prepare_root.append(centroid_dict[int(node)])

    prepare_root = np.array(prepare_root)
    centroide = (
        sum(prepare_root[:, 0]) / len(prepare_root[:, 0]),
        sum(prepare_root[:, 1]) / len(prepare_root[:, 1]),
    )
    centroid_dict[9999] = np.array(centroide)

    labels = nx.get_edge_attributes(H_sub, "weight")

    for edge, _ in labels.items():

        dm = dm_list[order_big_dict[query_dict[edge[0]]]]
        sdm = sdm_list[order_big_dict[query_dict[edge[0]]]]

        weight = dm[order_dict[edge[0]], order_dict[edge[1]]] * w + sdm[
            order_dict[edge[0]], order_dict[edge[1]]
        ] * (1 - w)
        H_sub[edge[0]][edge[1]]["weight"] = weight
    # tmp = H_sub

    H_sub = nx.algorithms.tree.minimum_spanning_arborescence(H_sub)
    H_nodes = list(range(len(H_sub.nodes)))

    node_convert = {}
    for pair in zip(list(H_sub.nodes), H_nodes):
        node_convert[pair[1]] = pair[0]

    adata.uns["PTS_graph"] = {}
    adata.uns["PTS_graph"]["graph"] = nx.to_scipy_sparse_array(H_sub)
    adata.uns["PTS_graph"]["node_dict"] = node_convert

    if return_graph:
        return H_sub


########################
## Global level PTS ##
########################


def get_node(node_list, split_node):
    result = np.array([])
    for node in node_list:
        result = np.append(result, np.array(split_node[node]).astype(int))
    return result.astype(int)


def ordering_nodes(node_list, use_label, adata):
    mean_dpt = []
    for node in node_list:
        mean_dpt.append(
            adata.obs[adata.obs[use_label] == str(node)]["dpt_pseudotime"].median()
        )

    return list(np.array(node_list)[np.argsort(mean_dpt)])


# def dpt_distance_matrix(adata, cluster1, cluster2, use_label):
#     tmp = adata.obs[adata.obs[use_label] == str(cluster1)]
#     chosen_adata1 = adata[list(tmp.index)]
#     tmp = adata.obs[adata.obs[use_label] == str(cluster2)]
#     chosen_aadata = adata[list(tmp.index)]

#     sub_dpt1 = []
#     chosen_sub1 = chosen_adata1.obs["sub_cluster_labels"].unique()
#     for i in range(0, len(chosen_sub1)):
#         sub_dpt1.append(
#             chosen_adata1.obs[
#                 chosen_adata1.obs["sub_cluster_labels"] == chosen_sub1[i]
#             ]["dpt_pseudotime"].median()
#         )

#     sub_dpt2 = []
#     chosen_sub2 = chosen_aadata.obs["sub_cluster_labels"].unique()
#     for i in range(0, len(chosen_sub2)):
#         sub_dpt2.append(
#             chosen_aadata.obs[
#                 chosen_aadata.obs["sub_cluster_labels"] == chosen_sub2[i]
#             ]["dpt_pseudotime"].median()
#         )

#     dm = cdist(
#         np.array(sub_dpt1).reshape(-1, 1),
#         np.array(sub_dpt2).reshape(-1, 1),
#         lambda u, v: v - u,
#     )
#     from sklearn.preprocessing import MinMaxScaler
#     scaler = MinMaxScaler()
#     scale_dm = scaler.fit_transform(dm)
#     # scale_dm = (dm + (-np.min(dm))) / np.max(dm)
#     return scale_dm


def spatial_distance_matrix(adata, cluster1, cluster2, use_label):

    tmp = adata.obs[adata.obs[use_label] == str(cluster1)]
    chosen_adata1 = adata[list(tmp.index)]
    tmp = adata.obs[adata.obs[use_label] == str(cluster2)]
    chosen_aadata = adata[list(tmp.index)]

    centroid_dict = adata.uns["centroid_dict"]
    centroid_dict = {int(key): centroid_dict[key] for key in centroid_dict}

    sub_coord1 = []
    chosen_sub1 = chosen_adata1.obs["sub_cluster_labels"].unique()
    for i in range(0, len(chosen_sub1)):
        sub_coord1.append(centroid_dict[int(chosen_sub1[i])])

    sub_coord2 = []
    chosen_sub2 = chosen_aadata.obs["sub_cluster_labels"].unique()
    for i in range(0, len(chosen_sub2)):
        sub_coord2.append(centroid_dict[int(chosen_sub2[i])])

    sdm = cdist(np.array(sub_coord1), np.array(sub_coord2), "euclidean")

    from sklearn.preprocessing import MinMaxScaler

    # scaler = MinMaxScaler()
    # scale_sdm = scaler.fit_transform(sdm)
    scale_sdm = sdm / np.max(sdm)

    return scale_sdm


def ge_distance_matrix(adata, cluster1, cluster2, use_label, use_rep, n_dims):

    tmp = adata.obs[adata.obs[use_label] == str(cluster1)]
    chosen_adata1 = adata[list(tmp.index)]
    tmp = adata.obs[adata.obs[use_label] == str(cluster2)]
    chosen_aadata = adata[list(tmp.index)]

    centroid_dict = adata.uns["centroid_dict"]
    centroid_dict = {int(key): centroid_dict[key] for key in centroid_dict}

    sub_coord1 = []
    chosen_sub1 = chosen_adata1.obs["sub_cluster_labels"].unique()
    for i in chosen_sub1:
        sub_coord1.append(
            np.array(
                chosen_adata1[chosen_adata1.obs["sub_cluster_labels"].isin([i])].obsm[
                    use_rep
                ][:, :n_dims]
            )
        )

    sub_coord2 = []
    chosen_sub2 = chosen_aadata.obs["sub_cluster_labels"].unique()
    for i in chosen_sub2:
        sub_coord2.append(
            np.array(
                chosen_aadata[chosen_aadata.obs["sub_cluster_labels"].isin([i])].obsm[
                    use_rep
                ][:, :n_dims]
            )
        )

    results = []
    for i in range(0, len(sub_coord1)):
        for j in range(0, len(sub_coord2)):
            results.append(cdist(sub_coord1[i], sub_coord2[j], "cosine").mean())
    results = np.array(results).reshape(len(sub_coord1), len(sub_coord2))

    from sklearn.preprocessing import MinMaxScaler

    # scaler = MinMaxScaler()
    # scale_sdm = scaler.fit_transform(results)
    scale_sdm = results / np.max(results)

    return scale_sdm


# def _density_normalize(other: Union[np.ndarray, spmatrix]
#     ) -> Union[np.ndarray, spmatrix]:
#         """
#         Density normalization by the underlying KNN graph.
#         Parameters
#         ----------
#         other:
#             Matrix to normalize.
#         Returns
#         -------
#         :class:`np.ndarray` or :class:`scipy.sparse.spmatrix`
#             Density normalized transition matrix.
#         """

#         q = np.asarray(_conn.sum(axis=0))

#         if not issparse(other):
#             Q = np.diag(1.0 / q)
#         else:
#             Q = spdiags(1.0 / q, 0, other.shape[0], other.shape[0])

#         return Q @ other @ Q


# def compute_transition_matrix(
#     adata: AnnData,
#     ):

# def compute_spatial_trans_probs(
#     adata: AnnData,
#     ):
