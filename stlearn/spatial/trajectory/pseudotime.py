import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from natsort import natsorted
from networkx import Graph
from sklearn.neighbors import NearestCentroid

from stlearn.spatial.clustering import localization
from stlearn.spatial.morphology import adjust
from stlearn.types import _METHOD


def pseudotime(
    adata: AnnData,
    use_label: str = "leiden",
    eps: float = 20,
    threshold: float = 0.01,
    radius: int = 50,
    method: _METHOD = "mean",
    threshold_spots: int = 5,
    use_sme: bool = False,
    reverse: bool = False,
    pseudotime_key: str = "dpt_pseudotime",
    max_nodes: int = 4,
    copy: bool = False,
) -> AnnData | None:
    """\
    Perform pseudotime analysis. Requires having run knn neighbours beforehand.

    Parameters
    ----------
    adata:
        Annotated data matrix.
    use_label:
        Use label result of cluster method.
    eps:
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other. This is not a maximum bound on the
        distances of points within a cluster. This is the most important DBSCAN
        parameter to choose appropriately for your data set and distance function.
    threshold:
        Threshold to find the significant connection for PAGA graph.
    radius:
        radius to adjust data for diffusion map
    method:
        method to adjust the data.
    use_sme:
        Use adjusted feature by sme normalization or not
    reverse:
        Reverse the pseudotime score
    pseudotime_key:
        Key to store pseudotime
    max_nodes:
        Maximum number of node in available paths
    copy:
        Return a copy instead of writing to adata.

    Notes
    -----
    Each run clears any previously computed values for: X_diffmap,
    X_draw_graph_fr, X_diffmap_morphology, split_node, global_graph,
    centroid_dict, available_paths, threshold_spots, and sub_cluster_labels.

    Returns
    -------
    Anndata

    """

    if "neighbors" not in adata.uns and "connectivities" not in adata.obsp:
        raise ValueError(
            "A neighbor graph is required - none found in uns or obsp. "
            "Subsetting data requires re-running."
        )

    keys_obsm = ["X_diffmap", "X_draw_graph_fr", "X_diffmap_morphology"]
    keys_uns = [
        "split_node",
        "global_graph",
        "centroid_dict",
        "available_paths",
        "threshold_spots",
    ]
    keys_obs = ["sub_cluster_labels"]

    for key in keys_obsm:
        adata.obsm.pop(key, None)
    for key in keys_uns:
        adata.uns.pop(key, None)
    for key in keys_obs:
        if key in adata.obs.columns:
            del adata.obs[key]

    localization(adata, use_label=use_label, eps=eps)

    # Running paga
    sc.tl.paga(adata, groups=use_label)

    # Denoising the graph
    sc.tl.diffmap(adata)

    if use_sme:
        adjust(adata, use_data="X_diffmap", radius=radius, method=method)
        adata.obsm["X_diffmap"] = adata.obsm["X_diffmap_morphology"]

    # Get connection matrix
    cnt_matrix = adata.uns["paga"]["connectivities"].toarray()

    # Filter by threshold
    cnt_matrix[cnt_matrix < threshold] = 0.0

    # Subclusters of each cluster, keyed by the cluster's position in the
    # categories, which is also its row in the PAGA matrix.
    split_node = {}
    for position, label in enumerate(adata.obs[use_label].cat.categories):
        meaningful_sub = []
        for i in adata.obs[adata.obs[use_label] == label][
            "sub_cluster_labels"
        ].unique():
            if len(adata.obs[adata.obs["sub_cluster_labels"] == i]) > threshold_spots:
                meaningful_sub.append(i)

        split_node[position] = meaningful_sub

    adata.uns["threshold_spots"] = threshold_spots
    # Stored with the positions as strings: anndata only saves string keys.
    adata.uns["split_node"] = {str(k): v for k, v in split_node.items()}

    # Connection matrix between subclusters, in label order: each subcluster
    # takes its cluster's row and column of the PAGA matrix.
    cluster_of = {
        subcluster: position
        for position, subclusters in split_node.items()
        for subcluster in subclusters
    }
    subclusters = natsorted(cluster_of)
    rows = [cluster_of[subcluster] for subcluster in subclusters]
    subcluster_matrix = pd.DataFrame(
        cnt_matrix[np.ix_(rows, rows)], index=subclusters, columns=subclusters
    )

    # Create a connection graph of subclusters
    graph = nx.from_pandas_adjacency(subcluster_matrix)

    adata.uns["global_graph"] = {
        "graph": nx.to_scipy_sparse_array(graph),
        "nodes": list(graph.nodes),
    }

    # Create centroid dict for subclusters
    clf = NearestCentroid()
    clf.fit(adata.obs[["imagecol", "imagerow"]].values, adata.obs["sub_cluster_labels"])
    centroid_dict = dict(zip(clf.classes_.tolist(), clf.centroids_, strict=True))

    def closest_node(node, nodes):
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - node) ** 2, axis=1)
        return np.argmin(dist_2)

    for cl in adata.obs["sub_cluster_labels"].unique():
        cl_points = adata.obs[adata.obs["sub_cluster_labels"] == cl][
            ["imagecol", "imagerow"]
        ].values
        new_centroid = cl_points[closest_node(centroid_dict[cl], cl_points)]
        centroid_dict[cl] = new_centroid

    adata.uns["centroid_dict"] = centroid_dict

    # Running diffusion pseudo-time
    sc.tl.dpt(adata)

    if reverse:
        adata.obs[pseudotime_key] = 1 - adata.obs[pseudotime_key]

    store_available_paths(adata, threshold, use_label, max_nodes, pseudotime_key)

    return adata if copy else None


def store_available_paths(adata, threshold, use_label, max_nodes, pseudotime_key):
    # Recreate original PAGA graph.
    graph = nx.from_numpy_array(adata.uns["paga"]["connectivities"].toarray())
    edge_weights = nx.get_edge_attributes(graph, "weight")
    graph.remove_edges_from((e for e, w in edge_weights.items() if w < threshold))

    # Calculate pseudotime for each node
    node_pseudotime = node_pseudotime_summary(adata, graph, pseudotime_key, use_label)

    # Convert undirected graph to directed graph by pseudotime.
    directed_graph = orient_by_pseudotime(graph, node_pseudotime)

    # Paths are stored as cluster labels. The keys use the clusters' positions:
    # labels can hold characters, like "/", that anndata can't save in keys.
    categories = adata.obs[use_label].cat.categories
    all_paths = {}

    for source in directed_graph.nodes:
        for target in directed_graph.nodes:
            paths = nx.all_simple_paths(
                directed_graph,
                source=source,
                target=target,
            )
            for i, path in enumerate(paths):
                if len(path) < max_nodes:
                    all_paths[str(i) + "_" + str(source) + "_" + str(target)] = [
                        categories[node] for node in path
                    ]

    adata.uns["available_paths"] = all_paths
    print(
        "All available trajectory paths are stored in adata.uns['available_paths'] "
        + "with length < "
        + str(max_nodes)
        + " nodes",
    )


def node_pseudotime_summary(adata, graph: Graph, pseudotime_key, use_label):
    # Graph nodes are PAGA rows, i.e. positions in the categories.
    categories = adata.obs[use_label].cat.categories
    summary = {}
    for node in graph.nodes:
        s = adata.obs.loc[adata.obs[use_label] == categories[node], pseudotime_key]
        finite = s[np.isfinite(s)]
        summary[node] = float(finite.median()) if len(finite) else np.nan
    return summary


def orient_by_pseudotime(graph, node_pseudotime):
    """Orient an undirected PAGA graph into a DAG using per-node pseudotime.

    Each undirected edge becomes a single arc pointing from lower to higher
    pseudotime. Edges touching a node with NaN pseudotime (a cluster
    unreachable from the root) cannot be ordered and are dropped. Ties are
    broken deterministically by node id so no 2-cycle can survive.
    """
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(graph.nodes)
    for u, v in graph.edges:
        pu, pv = node_pseudotime[u], node_pseudotime[v]
        if not (np.isfinite(pu) and np.isfinite(pv)):
            continue
        if pu < pv:
            directed_graph.add_edge(u, v)
        elif pv < pu:
            directed_graph.add_edge(v, u)
        else:
            directed_graph.add_edge(min(u, v), max(u, v))
    return directed_graph
