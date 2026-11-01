import networkx
import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.spatial.distance import cdist

from stlearn.utils import _TRAJECTORY_ROOT, _cluster_positions, _read_graph


def global_level(
    adata: AnnData,
    list_clusters: str | list[str],
    w: float,
    use_label: str = "leiden",
    use_rep: str = "X_pca",
    n_dims: int = 40,
    return_graph: bool = False,
    verbose: bool = True,
) -> networkx.Graph | None:
    """\
    Perform global sptial trajectory inference.

    Parameters
    ----------
    adata
        Annotated data matrix.
    list_clusters
        Labels of the clusters in ``adata.obs[use_label]`` to perform
        pseudo-space-time between, e.g. ``["6", "9"]``. Empty uses all clusters.
    w
        Pseudo-spatio-temporal distance weight (balance between spatial effect and DPT)
    use_label
        Use label result of cluster method.
    return_graph
        Return PTS graph
    Returns
    -------
    networkx.Graph:

    adata.uns["PTS_graph"]["graph"]:
    adata.uns["PTS_graph"]["nodes"]:
    """

    assert w <= 1, "w should be in range 0 to 1"
    # Get global graph
    G = _read_graph(adata, "global_graph")
    # Convert to directed graph
    H = G.to_directed()

    # Query clusters, as positions in the categories (PAGA rows). inds_cat gives
    # the label of each position.
    categories = adata.obs[use_label].cat.categories
    inds_cat = dict(enumerate(categories))
    query_nodes = _cluster_positions(adata, use_label, list_clusters)
    if len(query_nodes) == 0:
        print("No clusters specified, using all available clusters")
        query_nodes = list(inds_cat)

    query_nodes = ordering_nodes(query_nodes, use_label, adata)
    if verbose:
        print(
            "Start to construct the trajectory: "
            + " -> ".join(inds_cat[node] for node in query_nodes),
        )

    query_dict = {}
    order_dict = {}

    for i in query_nodes:
        for order, j in enumerate(
            adata.obs[adata.obs[use_label] == inds_cat[i]][
                "sub_cluster_labels"
            ].unique(),
        ):
            query_dict[j] = int(i)
            order_dict[j] = order
    dm_list = []
    sdm_list = []
    order_big_dict = {}
    edge_list = []

    split_node = adata.uns["split_node"]

    empty = [inds_cat[node] for node in query_nodes if len(split_node[str(node)]) == 0]
    if empty:
        raise ValueError(
            f"Cluster(s) {empty} have no subclusters of more than "
            f"{adata.uns['threshold_spots']} spots, so no trajectory can pass "
            "through them. Choose other clusters, or lower threshold_spots in "
            "stlearn.spatial.trajectory.pseudotime."
        )

    for i, j in enumerate(query_nodes):
        order_big_dict[j] = int(i)
        if i == len(query_nodes) - 1:
            break
        for j in split_node[str(query_nodes[i])]:
            for k in split_node[str(query_nodes[i + 1])]:
                edge_list.append((j, k))

        # Calculate DPT distance matrix
        dm_list.append(
            ge_distance_matrix(
                adata,
                inds_cat[query_nodes[i]],
                inds_cat[query_nodes[i + 1]],
                use_label=use_label,
                use_rep=use_rep,
                n_dims=n_dims,
            ),
        )
        # Calculate Spatial distance matrix
        sdm_list.append(
            spatial_distance_matrix(
                adata,
                inds_cat[query_nodes[i]],
                inds_cat[query_nodes[i + 1]],
                use_label=use_label,
            ),
        )

    H_sub = nx.DiGraph(H.edge_subgraph(edge_list))
    if not nx.is_connected(H_sub.to_undirected()):
        raise ValueError(
            "The chosen clusters are not available to construct the spatial "
            + "trajectory! Please choose other path.",
        )

    # The root of the trajectory tree points at every subcluster of the first
    # cluster, so the arborescence below starts there.
    for node in split_node[str(query_nodes[0])]:
        H_sub.add_edge(_TRAJECTORY_ROOT, node)

    labels = nx.get_edge_attributes(H_sub, "weight")

    for edge, _ in labels.items():
        dm = dm_list[order_big_dict[query_dict[edge[0]]]]
        sdm = sdm_list[order_big_dict[query_dict[edge[0]]]]

        weight = dm[order_dict[edge[0]], order_dict[edge[1]]] * w + sdm[
            order_dict[edge[0]], order_dict[edge[1]]
        ] * (1 - w)
        H_sub[edge[0]][edge[1]]["weight"] = weight

    # Set edges with weight=None to weight=0
    for u, v, tmp in H_sub.edges(data=True):
        if tmp.get("weight") is None:
            H_sub[u][v]["weight"] = 0

    H_sub = nx.algorithms.tree.minimum_spanning_arborescence(H_sub)

    adata.uns["PTS_graph"] = {
        "graph": nx.to_scipy_sparse_array(H_sub),
        "nodes": list(H_sub.nodes),
    }

    if return_graph:
        return H_sub
    else:
        return None


# Global level PTS
def ordering_nodes(node_list, use_label, adata):
    # node_list holds positions in the categories; order them by pseudotime.
    categories = adata.obs[use_label].cat.categories
    mean_dpt = []
    for node in node_list:
        mean_dpt.append(
            adata.obs[adata.obs[use_label] == categories[node]][
                "dpt_pseudotime"
            ].median(),
        )

    return list(np.array(node_list)[np.argsort(mean_dpt)])


def spatial_distance_matrix(adata, cluster1, cluster2, use_label):
    centroid_dict = adata.uns["centroid_dict"]
    _, _, chosen_sub1 = _subclusters(adata, cluster1, use_label)
    _, _, chosen_sub2 = _subclusters(adata, cluster2, use_label)
    sub_coord1 = [centroid_dict[sub] for sub in chosen_sub1]
    sub_coord2 = [centroid_dict[sub] for sub in chosen_sub2]

    sdm = cdist(np.array(sub_coord1), np.array(sub_coord2), "euclidean")

    return sdm / np.max(sdm)


def _subclusters(adata, cluster, use_label):
    """Rows of `cluster` and its subclusters, in the order they first appear."""
    in_cluster = (adata.obs[use_label] == str(cluster)).to_numpy()
    codes, subclusters = pd.factorize(
        adata.obs["sub_cluster_labels"].to_numpy()[in_cluster]
    )
    return np.flatnonzero(in_cluster), codes, subclusters


def ge_distance_matrix(adata, cluster1, cluster2, use_label, use_rep, n_dims):
    """Mean cosine distance between the spots of each pair of subclusters, in
    `use_rep`, scaled so the largest is 1.

    The mean over all pairs of 1 - u.v / (|u| |v|) is 1 - (mean of the unit
    vectors u / |u|) . (mean of the unit vectors v / |v|), so it is computed
    from each subcluster's mean unit vector, without a spots x spots matrix.
    """
    mean1 = _mean_unit_vectors(adata, cluster1, use_label, use_rep, n_dims)
    mean2 = _mean_unit_vectors(adata, cluster2, use_label, use_rep, n_dims)
    results = 1 - mean1 @ mean2.T
    return results / np.max(results)


def _mean_unit_vectors(adata, cluster, use_label, use_rep, n_dims):
    """Mean of the unit vectors of each subcluster's spots in `use_rep`."""
    rows, codes, subclusters = _subclusters(adata, cluster, use_label)
    vectors = np.asarray(adata.obsm[use_rep][:, :n_dims][rows], dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        units = vectors / norms  # a zero vector gives NaN, as cdist does
    sums = np.zeros((len(subclusters), units.shape[1]))
    np.add.at(sums, codes, units)
    return sums / np.bincount(codes, minlength=len(subclusters))[:, None]
