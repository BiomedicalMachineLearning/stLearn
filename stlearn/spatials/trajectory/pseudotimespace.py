from anndata import AnnData
from typing import Optional, Union
import numpy as np
import pandas as pd
import networkx as nx

def pseudotimespace(
    adata: AnnData,
    use_labels: str = "louvain",
    list_cluster: list = [],
    w: float = 0.6,
    method: str = "max",
    copy: bool = False,
) -> Optional[AnnData]:
    
    # Get global graph
    G = adata.uns["global_graph"]
    # Convert to directed graph
    H = G.to_directed()

    # Query cluster
    query_nodes = list_cluster
    query_nodes = ordering_nodes(query_nodes,adata)
    print(query_nodes)



    edge_list = []
    for i,j in enumerate(query_nodes):
        if i == len(query_nodes)-1:
            break
        for j in adata.uns["split_node"][query_nodes[i]]:
            for k in adata.uns["split_node"][query_nodes[i+1]]:
                edge_list.append((int(j),int(k)))

    # Get centroid dictionary
    centroid_dict=adata.uns["centroid_dict"]



    H_sub = H.edge_subgraph(edge_list)
    H_sub = nx.DiGraph(H_sub)
    prepare_root = []
    for node in adata.uns["split_node"][query_nodes[0]]:
        H_sub.add_edge(9999,int(node))
        prepare_root.append(centroid_dict[int(node)])

    prepare_root = np.array(prepare_root)
    centroide = (sum(prepare_root[:,0])/len(prepare_root[:,0]),sum(prepare_root[:,1])/len(prepare_root[:,1]))
    centroid_dict[9999] = centroide


    dpt_distance_dict = {}
    for i in adata.obs["sub_cluster_labels"].unique():
        if method == "max":
            dpt_distance_dict[i] = adata.obs[adata.obs["sub_cluster_labels"]==i]["dpt_pseudotime"].max()
        elif method == "mean":
            dpt_distance_dict[i] = adata.obs[adata.obs["sub_cluster_labels"]==i]["dpt_pseudotime"].mean()
        else:
            dpt_distance_dict[i] = adata.obs[adata.obs["sub_cluster_labels"]==i]["dpt_pseudotime"].median()

    centroid_scale_dict = {}
    x = []
    for i,j in centroid_dict.items():
        x.append(j)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scale = scaler.fit_transform(x)

    for i,j in enumerate(centroid_dict.items()):
        centroid_scale_dict[j[0]]=scale[i]


    labels = nx.get_edge_attributes(H_sub,'weight')

    for edge,_ in labels.items():
        H_sub[edge[0]][edge[1]]['weight'] = (np.linalg.norm(centroid_scale_dict[edge[1]]-
                                            centroid_scale_dict[edge[0]]))*w + (np.absolute(dpt_distance_dict[str(edge[0])] -
                                                                                dpt_distance_dict[str(edge[1])]))*(1-w)


    H_sub = nx.algorithms.tree.minimum_spanning_arborescence(H_sub)
    #remove = [edge for edge in H_sub.edges if 9999 in edge]
    #H_sub.remove_edges_from(remove)
    #remove.remove_node(9999)

    adata.uns["PTS_graph"] = H_sub







def get_node(node_list,split_node):
    result = np.array([])
    for node in node_list:
        result = np.append(result,np.array(split_node[node]).astype(int))
    return result.astype(int)

def ordering_nodes(node_list,adata):
    mean_dpt = []
    for node in node_list:
        mean_dpt.append(adata.obs[adata.obs["louvain"]==str(node)]["dpt_pseudotime"].mean())

    return list(np.array(node_list)[np.argsort(mean_dpt)])