import networkx
import numpy
from anndata import AnnData
from matplotlib import pyplot as plt
from numpy._typing import NDArray

from stlearn.pl.utils import (
    add_cluster_legend,
    draw_cluster_nodes,
    get_colors,
    place_legend_right_of_colour_bar,
    subclusters_of,
)
from stlearn.utils import _cluster_labels, _read_graph, _read_trajectory


def pseudotime_plot(
    adata: AnnData,
    library_id: str | None = None,
    use_label: str = "leiden",
    pseudotime_key: str = "dpt_pseudotime",
    list_clusters: str | list[str] | None = None,
    cell_alpha: float = 1.0,
    image_alpha: float = 1.0,
    edge_alpha: float = 0.8,
    node_alpha: float = 1.0,
    spot_size: float | int = 6.5,
    node_size: float = 60,
    show_color_bar: bool = True,
    show_axis: bool = False,
    show_graph: bool = True,
    show_trajectories: bool = False,
    reverse: bool = False,
    show_node: bool = True,
    show_legend: bool = True,
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
        Area of the nodes in points^2, as for matplotlib's scatter.
    show_color_bar
        Show color bar or not.
    show_axis
        Show axis or not.
    show_graph
        Show PAGA graph or not.
    show_trajectories
        Show the spatial trajectories or not.
    reverse
        Reverse the direction of the trajectories.
    show_node
        Show the nodes, one per subcluster, in their cluster's colour.
    show_legend
        Show the legend of cluster labels for the nodes, outside the plot.
    show_plot
        Show the plot or not.
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
        checked_list_clusters = list(adata.obs[use_label].cat.categories)
    else:
        checked_list_clusters = _cluster_labels(adata, use_label, list_clusters)

    imagecol = adata.obs["imagecol"]
    imagerow = adata.obs["imagerow"]
    tmp = adata.obs
    G = _read_graph(adata, "global_graph")

    result = []
    query_node = subclusters_of(adata, use_label, checked_list_clusters)
    for edge in G.edges(query_node):
        if (edge[0] in query_node) and (edge[1] in query_node):
            result.append(edge)

    result2 = [G.edges[edge]["weight"] + 0.5 for edge in result]

    if ax is None:
        fig, a = plt.subplots()
    else:
        a = ax
        fig = a.figure

    centroid_dict: dict[str, NDArray[numpy.float64]] = adata.uns["centroid_dict"]
    dpt = adata.obs[pseudotime_key]
    vmin = min(dpt)
    vmax = max(dpt)
    # Plot scatter plot based on pixel of spots
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scale = scaler.fit_transform(tmp[pseudotime_key].values.reshape(-1, 1)).reshape(
        -1,
        1,
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
    if show_graph:
        networkx.draw_networkx_edges(
            G,
            pos=centroid_dict,
            ax=a,
            edgelist=result,
            width=result2,
            alpha=edge_alpha,
            edge_color="#333333",
        )
    if show_trajectories:
        if "PTS_graph" not in adata.uns:
            raise ValueError("Please run stlearn.spatial.trajectory.pseudotimespace!")

        G = _read_trajectory(adata)

        networkx.draw_networkx_edges(
            G,
            pos=centroid_dict,
            ax=a,
            node_size=node_size,
            alpha=1.0,
            width=2.5,
            edge_color="#f4efd3",
            arrowsize=17,
            arrowstyle="<|-" if reverse else "-|>",
            connectionstyle="arc3,rad=0.2",
        )

    if show_color_bar:
        cb = fig.colorbar(plot, ax=a)
        cb.outline.set_visible(False)

    # Nodes take their cluster's colour; the legend names the clusters.
    if show_node and (show_graph or show_trajectories):
        colors = get_colors(adata, use_label)
        drawn = draw_cluster_nodes(
            a,
            adata,
            use_label,
            checked_list_clusters,
            colors,
            size=node_size,
            alpha=node_alpha,
        )
        if show_legend:
            categories = adata.obs[use_label].cat.categories
            legend = add_cluster_legend(
                a,
                [categories[p] for p in drawn],
                [colors[p] for p in drawn],
            )
            if show_color_bar:
                # Beyond the colour bar's tick labels, which sit outside its axes.
                place_legend_right_of_colour_bar(legend, a, cb.ax)

    if library_id is None:
        library_id = next(iter(adata.uns["spatial"].keys()))

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
