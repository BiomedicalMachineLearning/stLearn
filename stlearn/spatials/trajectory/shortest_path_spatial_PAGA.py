import networkx as nx
import numpy as np

from stlearn.utils import _read_graph


def shortest_path_spatial_PAGA(
    adata,
    use_label,
    key="dpt_pseudotime",
):
    # Read original PAGA graph
    G = nx.from_numpy_array(adata.uns["paga"]["connectivities"].toarray())
    edge_weights = nx.get_edge_attributes(G, "weight")
    G.remove_edges_from((e for e, w in edge_weights.items() if w < 0))
    H = G.to_directed()

    # Get min_node and max_node
    min_node, max_node = find_min_max_node(adata, key, use_label)

    # Calculate pseudotime for each node
    node_pseudotime = {}

    for node in H.nodes:
        node_pseudotime[node] = adata.obs.query(use_label + " == '" + str(node) + "'")[
            key
        ].max()

    # Force original PAGA to directed PAGA based on pseudotime
    edge_to_remove = []
    for edge in H.edges:
        if node_pseudotime[edge[0]] - node_pseudotime[edge[1]] > 0:
            edge_to_remove.append(edge)
    H.remove_edges_from(edge_to_remove)

    # Extract all available paths
    all_paths = {}
    j = 0
    for source in H.nodes:
        for target in H.nodes:
            paths = nx.all_simple_paths(H, source=source, target=target)
            for i, path in enumerate(paths):
                j += 1
                all_paths[j] = path

    # Filter the target paths from min_node to max_node
    target_paths = []
    for path in list(all_paths.values()):
        if path[0] == min_node and path[-1] == max_node:
            target_paths.append(path)

    # Get the global graph
    G = _read_graph(adata, "global_graph")

    centroid_dict = adata.uns["centroid_dict"]
    centroid_dict = {int(key): centroid_dict[key] for key in centroid_dict}

    # Generate total length of every path. Store by dictionary
    dist_dict = {}
    for path in target_paths:
        path_name = ",".join(list(map(str, path)))
        result = []
        query_node = get_node(path, adata.uns["split_node"])
        for edge in G.edges():
            if (edge[0] in query_node) and (edge[1] in query_node):
                result.append(edge)
        if len(result) >= len(path):
            dist_dict[path_name] = calculate_total_dist(result, centroid_dict)

    # Find the shortest path
    shortest_path = min(dist_dict, key=lambda x: dist_dict[x])
    return shortest_path.split(",")


# get name of cluster by subcluster
def get_cluster(search, dictionary):
    for cl, sub in dictionary.items():
        if search in sub:
            return cl


def get_node(node_list, split_node):
    result = np.array([])
    for node in node_list:
        result = np.append(result, np.array(split_node[int(node)]).astype(int))
    return result.astype(int)


def find_min_max_node(adata, key="dpt_pseudotime", use_label="leiden"):
    min_cluster = int(adata.obs[adata.obs[key] == 0][use_label].values[0])
    max_cluster = int(adata.obs[adata.obs[key] == 1][use_label].values[0])

    return [min_cluster, max_cluster]


def calculate_total_dist(result, centroid_dict):
    import math

    total_dist = 0
    for edge in result:
        source = centroid_dict[edge[0]]
        target = centroid_dict[edge[1]]
        dist = math.dist(source, target)
        total_dist += dist
    return total_dist
