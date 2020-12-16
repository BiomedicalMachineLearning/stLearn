from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import matplotlib
import numpy as np
import networkx as nx
from stlearn._compat import Literal
from typing import Optional, Union
from anndata import AnnData
import warnings

# from .utils import get_img_from_fig, checkType


def cluster_plot(
    adata: AnnData,
    library_id: str = None,
    use_label: str = "louvain",
    list_cluster: list = None,
    data_alpha: float = 1.0,
    cmap: str = "default",
    tissue_alpha: float = 1.0,
    threshold_spots: int = 0,
    title: str = None,
    spot_size: Union[float, int] = 6.5,
    show_axis: bool = False,
    show_legend: bool = True,
    show_trajectory: bool = False,
    reverse: bool = False,
    show_subcluster: bool = False,
    show_node: bool = True,
    show_cluster_text: bool = False,
    cropped: bool = True,
    margin: int = 100,
    show_plot: bool = True,
    name: str = None,
    dpi: int = 150,
    output: str = None,
    copy: bool = False,
) -> Optional[AnnData]:

    """\
    Clustering plot for sptial transcriptomics data. Also it has a function to display trajectory inference.

    Parameters
    ----------
    adata
        Annotated data matrix.
    library_id
        Library id stored in AnnData.
    use_label
        Use label result of clustering method.
    list_cluster
        Choose set of clusters that will display in the plot.
    data_alpha
        Opacity of the spot.
    tissue_alpha
        Opacity of the tissue.
    cmap
        Color map to use.
    spot_size
        Size of the spot.
    show_axis
        Show axis or not.
    show_legend
        Show legend or not.
    show_trajectory
        Show the spatial trajectory or not. It requires stlearn.spatial.trajectory.pseudotimespace.
    show_subcluster
        Show subcluster or not. It requires stlearn.spatial.trajectory.global_level.
    name
        Name of the output figure file.
    dpi
        DPI of the output figure.
    output
        Save the figure as file or not.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Nothing
    """

    # plt.rcParams['figure.dpi'] = dpi

    n_clusters = len(adata.obs[use_label].unique())

    imagecol = adata.obs["imagecol"]
    imagerow = adata.obs["imagerow"]

    # Option for turning off showing figure
    plt.ioff()

    # Initialize matplotlib
    fig, a = plt.subplots()

    from scanpy.plotting import palettes
    from stlearn.plotting import palettes_st

    if cmap == "vega_10_scanpy":
        cmap = palettes.vega_10_scanpy
    elif cmap == "vega_20_scanpy":
        cmap = palettes.vega_20_scanpy
    elif cmap == "default_102":
        cmap = palettes.default_102
    elif cmap == "default_28":
        cmap = palettes.default_28
    elif cmap == "jana_40":
        cmap = palettes_st.jana_40
    elif cmap == "default":
        cmap = palettes_st.default
    else:
        raise ValueError(
            "We only support vega_10_scanpy, vega_20_scanpy, default_28, default_102"
        )

    cmaps = matplotlib.colors.LinearSegmentedColormap.from_list("", cmap)

    cmap_ = plt.cm.get_cmap(cmaps)

    if list_cluster is not None:
        tmp = adata.obs.loc[
            adata.obs[use_label].isin(np.array(list_cluster).astype(str))
        ]
    else:
        tmp = adata.obs

    # Plot scatter plot based on pixel of spots
    adata.uns[use_label + "_colors"] = []

    for i, cluster in enumerate(tmp.groupby(use_label)):

        _ = a.scatter(
            cluster[1]["imagecol"],
            cluster[1]["imagerow"],
            c=[cmap_(int(i) / (len(cmap) - 1))],
            label=cluster[0],
            edgecolor="none",
            alpha=data_alpha,
            s=spot_size,
            marker="o",
        )

        adata.uns[use_label + "_colors"].append(
            matplotlib.colors.to_hex(cmap_(int(i) / (len(cmap) - 1)))
        )

    if show_legend:
        lgnd = a.legend(
            bbox_to_anchor=(1.3, 1.0),
            labelspacing=0.05,
            fontsize=8,
            handleheight=1.0,
            edgecolor="white",
        )
        for handle in lgnd.legendHandles:
            handle.set_sizes([20.0])

    if show_trajectory:

        used_colors = adata.uns[use_label + "_colors"]
        cmaps = matplotlib.colors.LinearSegmentedColormap.from_list("", used_colors)

        cmap = plt.get_cmap(cmaps)

        if not adata.uns["PTS_graph"]:
            raise ValueError("Please run stlearn.spatial.trajectory.pseudotimespace!")

        tmp = adata.uns["PTS_graph"]

        G = tmp.copy()

        remove = [edge for edge in G.edges if 9999 in edge]
        G.remove_edges_from(remove)
        G.remove_node(9999)
        centroid_dict = adata.uns["centroid_dict"]
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
            for x, y in centroid_dict.items():

                if x in get_node(list_cluster, adata.uns["split_node"]):
                    a.text(
                        y[0],
                        y[1],
                        get_cluster(str(x), adata.uns["split_node"]),
                        color="black",
                        fontsize=8,
                        zorder=100,
                        bbox=dict(
                            facecolor=cmap(
                                int(get_cluster(str(x), adata.uns["split_node"]))
                                / (len(used_colors) - 1)
                            ),
                            boxstyle="circle",
                            alpha=1,
                        ),
                    )

    if not show_axis:
        a.axis("off")

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"][
        adata.uns["spatial"]["use_quality"]
    ]

    # Overlay the tissue image
    a.imshow(
        image,
        alpha=tissue_alpha,
        zorder=-1,
    )

    if show_subcluster:
        if "sub_cluster_labels" not in adata.obs.columns:
            raise ValueError("Please run stlearn.spatial.cluster.localization")

        for cluster in list_cluster:

            if (
                len(
                    adata.obs[adata.obs[use_label] == str(cluster)][
                        "sub_cluster_labels"
                    ].unique()
                )
                < 2
            ):
                centroids = [
                    centroidpython(
                        adata.obs[adata.obs["sub_cluster_labels"] == str(cluster)][
                            ["imagecol", "imagerow"]
                        ].values
                    )
                ]
                classes = np.array(
                    [
                        adata.obs[adata.obs[use_label] == str(cluster)][
                            "sub_cluster_labels"
                        ][0]
                    ]
                )

            else:
                from sklearn.neighbors import NearestCentroid

                clf = NearestCentroid()
                clf.fit(
                    adata.obs[adata.obs[use_label] == str(cluster)][
                        ["imagecol", "imagerow"]
                    ].values,
                    adata.obs[adata.obs[use_label] == str(cluster)][
                        "sub_cluster_labels"
                    ],
                )

                centroids = clf.centroids_
                classes = clf.classes_

            for i, label in enumerate(classes):
                if (
                    len(adata.obs[adata.obs["sub_cluster_labels"] == label])
                    > threshold_spots
                ):
                    if centroids[i][0] < 1500:
                        x = -100
                        y = 50
                    else:
                        x = 100
                        y = -50
                    a.text(
                        centroids[i][0] + x,
                        centroids[i][1] + y,
                        label,
                        color="black",
                        fontsize=5,
                        zorder=3,
                        bbox=dict(
                            facecolor=adata.uns[use_label + "_colors"][int(cluster)],
                            boxstyle="round",
                            alpha=1.0,
                        ),
                    )

    if show_cluster_text:
        if list_cluster is None:
            list_cluster = adata.obs[use_label].unique()

        for cluster in list_cluster:
            if (
                len(adata.obs[adata.obs[use_label] == str(cluster)][use_label].unique())
                < 2
            ):
                centroids = [
                    centroidpython(
                        adata.obs[adata.obs[use_label] == str(cluster)][
                            ["imagecol", "imagerow"]
                        ].values
                    )
                ]
                classes = np.array(
                    [adata.obs[adata.obs[use_label] == str(cluster)][use_label][0]]
                )

            else:
                from sklearn.neighbors import NearestCentroid

                clf = NearestCentroid()
                clf.fit(
                    adata.obs[adata.obs[use_label] == str(cluster)][
                        ["imagecol", "imagerow"]
                    ].values,
                    adata.obs[adata.obs[use_label] == str(cluster)][use_label],
                )

                centroids = clf.centroids_
                classes = clf.classes_

            for i, label in enumerate(classes):
                if len(adata.obs[adata.obs[use_label] == label]) > threshold_spots:
                    if centroids[i][0] < 1500:
                        x = -100
                        y = 50
                    else:
                        x = 100
                        y = -50
                    a.text(
                        centroids[i][0] + x,
                        centroids[i][1] + y,
                        label,
                        color="black",
                        fontsize=5,
                        zorder=3,
                        bbox=dict(
                            facecolor=adata.uns[use_label + "_colors"][int(cluster)],
                            boxstyle="round",
                            alpha=1.0,
                        ),
                    )

    if cropped:
        a.set_xlim(imagecol.min() - margin, imagecol.max() + margin)

        a.set_ylim(imagerow.min() - margin, imagerow.max() + margin)

        a.set_ylim(a.get_ylim()[::-1])
        # plt.gca().invert_yaxis()

    if name is None:
        name = use_label

    if output is not None:
        fig.savefig(output + "/" + name, dpi=dpi, bbox_inches="tight", pad_inches=0)

    if show_plot == True:
        plt.show()


def centroidpython(data):
    x, y = zip(*data)
    l = len(x)
    return sum(x) / l, sum(y) / l


def get_cluster(search, dictionary):
    for (
        cl,
        sub,
    ) in (
        dictionary.items()
    ):  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if search in sub:
            return cl


def get_node(node_list, split_node):
    result = np.array([])
    for node in node_list:
        result = np.append(result, np.array(split_node[node]).astype(int))
    return result.astype(int)
