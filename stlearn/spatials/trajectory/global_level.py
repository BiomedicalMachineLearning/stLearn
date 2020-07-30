from anndata import AnnData
from typing import Optional, Union
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist


def global_level(
    adata: AnnData,
    use_label: str = "louvain",
    list_cluster: list = [],
    w: float = 0.0,
    copy: bool = False,
) -> Optional[AnnData]:

    """\
    Perform global sptial trajectory inference.

    Parameters
    ----------
    adata
        Annotated data matrix.
    list_cluster
        Setup a list of cluster to perform pseudo-space-time
    use_label
        Use label result of clustering method.
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
    G = adata.uns["global_graph"]
    # Convert to directed graph
    H = G.to_directed()

    # Query cluster
    query_nodes = list_cluster
    query_nodes = ordering_nodes(query_nodes, use_label,adata)

    query_dict = {}
    order_dict = {}

    for i in query_nodes:
        order = 0
        for j in adata.obs[adata.obs[use_label] == str(i)]["sub_cluster_labels"].unique():
            if len(adata.obs[adata.obs["sub_cluster_labels"] == str(j)]) > adata.uns["threshold_spots"]:
                query_dict[int(j)] = int(i)
                order_dict[int(j)] = int(order)
                order += 1
    
    

    dm_list = []
    sdm_list = []
    order_big_dict = {}
    edge_list = []
    for i, j in enumerate(query_nodes):
        order_big_dict[j] = int(i)
        if i == len(query_nodes)-1:
            break
        for j in adata.uns["split_node"][query_nodes[i]]:
            for k in adata.uns["split_node"][query_nodes[i+1]]:
                edge_list.append((int(j), int(k)))

        # Calculate DPT distance matrix
        dm_list.append(dpt_distance_matrix(
            adata, query_nodes[i], query_nodes[i+1], use_label=use_label))
        # Calculate Spatial distance matrix
        sdm_list.append(spatial_distance_matrix(
            adata, query_nodes[i], query_nodes[i+1], use_label=use_label))
    

    # Get centroid dictionary
    centroid_dict = adata.uns["centroid_dict"]

    H_sub = H.edge_subgraph(edge_list)
    H_sub = nx.DiGraph(H_sub)
    prepare_root = []
    for node in adata.uns["split_node"][query_nodes[0]]:
        H_sub.add_edge(9999, int(node))
        prepare_root.append(centroid_dict[int(node)])

    prepare_root = np.array(prepare_root)
    centroide = (sum(prepare_root[:, 0])/len(prepare_root[:, 0]),
                 sum(prepare_root[:, 1])/len(prepare_root[:, 1]))
    centroid_dict[9999] = centroide

    labels = nx.get_edge_attributes(H_sub, 'weight')

    
    for edge, _ in labels.items():

        dm = dm_list[order_big_dict[query_dict[edge[0]]]]
        sdm = sdm_list[order_big_dict[query_dict[edge[0]]]]

        weight = dm[order_dict[edge[0]], order_dict[edge[1]]] * \
            w + sdm[order_dict[edge[0]], order_dict[edge[1]]]*(1-w)
        H_sub[edge[0]][edge[1]]['weight'] = weight

    H_sub = nx.algorithms.tree.minimum_spanning_arborescence(H_sub)
    #remove = [edge for edge in H_sub.edges if 9999 in edge]
    # H_sub.remove_edges_from(remove)
    # remove.remove_node(9999)

    adata.uns["PTS_graph"] = H_sub


def get_node(node_list, split_node):
    result = np.array([])
    for node in node_list:
        result = np.append(result, np.array(split_node[node]).astype(int))
    return result.astype(int)


def ordering_nodes(node_list,use_label, adata):
    max_dpt = []
    for node in node_list:
        max_dpt.append(adata.obs[adata.obs[use_label]
                                  == str(node)]["dpt_pseudotime"].max())

    return list(np.array(node_list)[np.argsort(max_dpt)])


def dpt_distance_matrix(adata, cluster1, cluster2, use_label):
    tmp = adata.obs[adata.obs[use_label] == str(cluster1)]
    chosen_adata1 = adata[list(tmp.index)]
    tmp = adata.obs[adata.obs[use_label] == str(cluster2)]
    chosen_adata2 = adata[list(tmp.index)]

    sub_dpt1 = []
    chosen_sub1 = chosen_adata1.obs["sub_cluster_labels"].unique()
    for i in range(0, len(chosen_sub1)):
        sub_dpt1.append(
            chosen_adata1.obs[chosen_adata1.obs["sub_cluster_labels"] == chosen_sub1[i]]["dpt_pseudotime"].max())

    sub_dpt2 = []
    chosen_sub2 = chosen_adata2.obs["sub_cluster_labels"].unique()
    for i in range(0, len(chosen_sub2)):
        sub_dpt2.append(
            chosen_adata2.obs[chosen_adata2.obs["sub_cluster_labels"] == chosen_sub2[i]]["dpt_pseudotime"].max())

    dm = cdist(np.array(sub_dpt1).reshape(-1, 1),
               np.array(sub_dpt2).reshape(-1, 1), lambda u, v: np.abs(u-v))
    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler()
    #scale_dm = scaler.fit_transform(dm)
    scale_dm = dm/np.max(dm)
    return scale_dm


def spatial_distance_matrix(adata, cluster1, cluster2, use_label):

    tmp = adata.obs[adata.obs[use_label] == str(cluster1)]
    chosen_adata1 = adata[list(tmp.index)]
    tmp = adata.obs[adata.obs[use_label] == str(cluster2)]
    chosen_adata2 = adata[list(tmp.index)]

    centroid_dict = adata.uns["centroid_dict"]

    sub_coord1 = []
    chosen_sub1 = chosen_adata1.obs["sub_cluster_labels"].unique()
    for i in range(0, len(chosen_sub1)):
        sub_coord1.append(centroid_dict[int(chosen_sub1[i])])

    sub_coord2 = []
    chosen_sub2 = chosen_adata2.obs["sub_cluster_labels"].unique()
    for i in range(0, len(chosen_sub2)):
        sub_coord2.append(centroid_dict[int(chosen_sub2[i])])

    sdm = cdist(np.array(sub_coord1), np.array(sub_coord2), "euclidean")

    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler()
    #scale_sdm = scaler.fit_transform(sdm)
    scale_sdm = sdm/np.max(sdm)

    return scale_sdm
