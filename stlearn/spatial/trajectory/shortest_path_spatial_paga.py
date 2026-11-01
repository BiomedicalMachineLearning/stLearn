import networkx as nx

from stlearn.utils import _read_graph


def shortest_path_spatial_paga(
    adata,
    use_label,
    key="dpt_pseudotime",
):
    """\
    Shortest path through the clusters, from the first cluster to the last.

    The first cluster is the one holding the spot where ``adata.obs[key]`` is 0
    (the root), and the last is the one holding the spot where it is 1. A single
    spot decides each, so the last cluster is not necessarily the one that comes
    last overall: ``pseudotimespace_global`` can order the same clusters
    differently, and the first and last cluster can be the same one.

    Candidate paths follow the PAGA graph, with each edge directed from the
    cluster with the lower maximum pseudotime to the higher. A path is kept only
    if its clusters' subclusters are connected in the global graph, and the
    shortest is the one whose global-graph edges between those subclusters have
    the smallest total length between centroids.

    Requires ``st.spatial.trajectory.pseudotime`` to have been run.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    use_label: str
        Column in ``adata.obs`` holding the cluster labels.
    key: str, default = "dpt_pseudotime"
        Column in ``adata.obs`` holding the pseudotime. It must be scaled so the
        earliest spot is exactly 0 and the latest exactly 1.

    Returns
    -------
    list of str
        Labels of the clusters on the shortest path, first to last.

    Raises
    ------
    ValueError
        If the first and last cluster are the same, or no path between them has
        its subclusters connected in the global graph.
    """
    # Read original PAGA graph
    G = nx.from_numpy_array(adata.uns["paga"]["connectivities"].toarray())
    edge_weights = nx.get_edge_attributes(G, "weight")
    G.remove_edges_from(e for e, w in edge_weights.items() if w < 0)
    H = G.to_directed()

    # PAGA nodes are positions in the categories.
    categories = adata.obs[use_label].cat.categories

    # Get min_node and max_node
    min_node, max_node = find_min_max_node(adata, key, use_label)
    if min_node == max_node:
        raise ValueError(
            f"The first and last spots in {key!r} are both in cluster "
            f"{categories[min_node]!r}, so there is no path between clusters."
        )

    # Calculate pseudotime for each node
    node_pseudotime = {
        node: adata.obs.loc[adata.obs[use_label] == categories[node], key].max()
        for node in H.nodes
    }

    # Force original PAGA to a directed PAGA based on pseudotime
    H.remove_edges_from(
        [(u, v) for u, v in H.edges if node_pseudotime[u] > node_pseudotime[v]]
    )

    # Paths through the directed PAGA from min_node to max_node
    target_paths = list(nx.all_simple_paths(H, source=min_node, target=max_node))

    # Get the global graph
    G = _read_graph(adata, "global_graph")
    centroid_dict = adata.uns["centroid_dict"]
    split_node = adata.uns["split_node"]

    # Total length of each path whose subclusters are connected in the global graph
    dist_dict = {}
    for path in target_paths:
        if not all(split_node[str(node)] for node in path):
            continue  # a cluster on the path has no subclusters
        query_node = [sub for node in path for sub in split_node[str(node)]]
        subgraph = G.subgraph(query_node)
        if nx.is_connected(subgraph.to_undirected()):
            result = list(subgraph.edges())
            dist_dict[tuple(path)] = calculate_total_dist(result, centroid_dict)

    if not dist_dict:
        raise ValueError(
            f"No path from {categories[min_node]!r} to {categories[max_node]!r} "
            "has its subclusters connected in the global graph."
        )

    # Find the shortest path, as cluster labels
    shortest_path = min(dist_dict, key=dist_dict.get)
    return [categories[node] for node in shortest_path]


def find_min_max_node(adata, key="dpt_pseudotime", use_label="leiden"):
    """Positions in the categories of the clusters holding the first and last spot.

    The first spot is where ``adata.obs[key]`` is exactly 0, and the last where it
    is exactly 1. A single spot decides each, so the cluster returned as last is
    not necessarily the one that comes last overall.
    """
    # Positions in the categories of the clusters holding the first and last spot.
    categories = adata.obs[use_label].cat.categories
    min_cluster = categories.get_loc(
        adata.obs[adata.obs[key] == 0][use_label].values[0]
    )
    max_cluster = categories.get_loc(
        adata.obs[adata.obs[key] == 1][use_label].values[0]
    )

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
