import matplotlib
import networkx as nx
import numpy as np
from anndata import AnnData
from matplotlib import pyplot as plt
from numpy._typing import NDArray

from stlearn.pl.utils import get_cluster, get_node
from stlearn.utils import _read_graph


def pseudotime_plot(
    adata: AnnData,
    library_id: str | None = None,
    use_label: str = "louvain",
    pseudotime_key: str = "dpt_pseudotime",
    list_clusters: str | list[str] | None = None,
    cell_alpha: float = 1.0,
    image_alpha: float = 1.0,
    edge_alpha: float = 0.8,
    node_alpha: float = 1.0,
    spot_size: float | int = 6.5,
    node_size: float = 5,
    show_color_bar: bool = True,
    show_axis: bool = False,
    show_graph: bool = True,
    show_trajectories: bool = False,
    reverse: bool = False,
    show_node: bool = True,
    show_plot: bool = True,
    cropped: bool = True,
    margin: int = 100,
    dpi: int = 150,
    output: str | None = None,
    name: str | None = None,
    ax=None,
) -> AnnData | None:
    """\
    Global trajectory inference plot (Only DPT).

    Parameters
    ----------
    adata
        Annotated data matrix.
    library_id
        Library id stored in AnnData.
    use_label
        Use label result of cluster method.
    list_clusters
        Choose set of clusters that will display in the plot.
    cell_alpha
        Opacity of the spot.
    image_alpha
        Opacity of the tissue.
    edge_alpha
        Opacity of edge in PAGA graph in the tissue.
    node_alpha
        Opacity of node in PAGA graph in the tissue.
    cmap
        Color map to use.
    spot_size
        Size of the spot.
    node_size
        Size of node in PAGA graph in the tissue.
    show_color_bar
        Show color bar or not.
    show_axis
        Show axis or not.
    show_graph
        Show PAGA graph or not.
    show_legend
        Show legend or not.
    show_plot
        Show plot or not
    dpi
        DPI of the output figure.
    output
        The output folder of the plot.
    name
        The filename of the plot.
    Returns
    -------
    Nothing
    """

    checked_list_clusters: list[str]
    if list_clusters is None:
        checked_list_clusters = adata.obs[use_label].cat.categories
    elif isinstance(list_clusters, str):
        checked_list_clusters = [list_clusters]
    else:
        checked_list_clusters = list_clusters

    imagecol = adata.obs["imagecol"]
    imagerow = adata.obs["imagerow"]
    tmp = adata.obs
    G = _read_graph(adata, "global_graph")

    labels = nx.get_edge_attributes(G, "weight")

    result = []
    query_node = list(
        map(int, get_node(checked_list_clusters, adata.uns["split_node"]))
    )
    for edge in G.edges(query_node):
        if (edge[0] in query_node) and (edge[1] in query_node):
            result.append(edge)

    result2 = []
    for edge in result:
        try:
            result2.append(labels[edge] + 0.5)
        except:
            result2.append(labels[edge[::-1]] + 0.5)

    fig, a = plt.subplots()
    if ax is not None:
        a = ax
    centroid_dict: dict[int, NDArray[np.float64]] = adata.uns["centroid_dict"]
    centroid_dict = {int(key): centroid_dict[key] for key in centroid_dict}
    dpt = adata.obs[pseudotime_key]
    vmin = min(dpt)
    vmax = max(dpt)
    # Plot scatter plot based on pixel of spots
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scale = scaler.fit_transform(tmp[pseudotime_key].values.reshape(-1, 1)).reshape(
        -1, 1
    )

    plot = a.scatter(
        tmp["imagecol"],
        tmp["imagerow"],
        edgecolor="none",
        alpha=cell_alpha,
        s=spot_size,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        cmap=plt.get_cmap("viridis"),
        c=scale.reshape(1, -1)[0],
    )
    used_colors = adata.uns[use_label + "_colors"]
    cmaps = matplotlib.colors.LinearSegmentedColormap.from_list("", used_colors)
    cmap = plt.get_cmap(cmaps)
    if show_graph:
        nx.draw_networkx(
            G,
            pos=centroid_dict,
            node_size=1,
            font_size=0,
            linewidths=2,
            edgelist=result,
            width=result2,
            alpha=edge_alpha,
            edge_color="#333333",
        )

        for node, pos in centroid_dict.items():
            if str(node) in get_node(checked_list_clusters, adata.uns["split_node"]):
                a.text(
                    pos[0],
                    pos[1],
                    get_cluster(str(node), adata.uns["split_node"]),
                    color="white",
                    fontsize=node_size,
                    zorder=100,
                    bbox=dict(
                        facecolor=cmap(
                            int(get_cluster(str(node), adata.uns["split_node"]))
                            / (len(used_colors) - 1)
                        ),
                        boxstyle="circle",
                        alpha=node_alpha,
                    ),
                )

    if show_trajectories:
        used_colors = adata.uns[use_label + "_colors"]
        cmaps = matplotlib.colors.LinearSegmentedColormap.from_list("", used_colors)

        cmap = plt.get_cmap(cmaps)

        if "PTS_graph" not in adata.uns:
            raise ValueError("Please run stlearn.spatial.trajectory.pseudotimespace!")

        tmp = _read_graph(adata, "PTS_graph")

        G = tmp.copy()

        remove = [edge for edge in G.edges if 9999 in edge]
        G.remove_edges_from(remove)
        G.remove_node(9999)
        centroid_dict = adata.uns["centroid_dict"]
        centroid_dict = {int(key): centroid_dict[key] for key in centroid_dict}
        if reverse:
            nx.draw_networkx_edges(
                G,
                pos=centroid_dict,
                node_size=10,
                alpha=1.0,
                width=2.5,
                edge_color="#f4efd3",
                arrowsize=17,
                arrowstyle="<|-",
                connectionstyle="arc3,rad=0.2",
            )
        else:
            nx.draw_networkx_edges(
                G,
                pos=centroid_dict,
                node_size=10,
                alpha=1.0,
                width=2.5,
                edge_color="#f4efd3",
                arrowsize=17,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.2",
            )

        if show_node:
            for node, pos in centroid_dict.items():
                if str(node) in get_node(
                    checked_list_clusters, adata.uns["split_node"]
                ):
                    a.text(
                        pos[0],
                        pos[1],
                        str(get_cluster(str(node), adata.uns["split_node"])),
                        color="black",
                        fontsize=8,
                        zorder=100,
                        bbox=dict(
                            facecolor=cmap(
                                get_cluster(str(node), adata.uns["split_node"])
                                / (len(used_colors) - 1)
                            ),
                            boxstyle="circle",
                            alpha=1,
                        ),
                    )

    if show_color_bar:
        cb = plt.colorbar(plot, cmap="viridis")
        cb.outline.set_visible(False)

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"][
        adata.uns["spatial"][library_id]["use_quality"]
    ]

    a.imshow(
        image,
        alpha=image_alpha,
        zorder=-1,
    )

    if not show_axis:
        a.axis("off")

    if cropped:
        a.set_xlim(imagecol.min() - margin, imagecol.max() + margin)

        a.set_ylim(imagerow.min() - margin, imagerow.max() + margin)

        a.set_ylim(a.get_ylim()[::-1])
        # plt.gca().invert_yaxis()

    if output is not None and name is not None:
        fig.savefig(output + "/" + name, dpi=dpi, bbox_inches="tight", pad_inches=0)
    if show_plot:
        plt.show()

    return adata
