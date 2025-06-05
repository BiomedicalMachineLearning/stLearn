import math
import random
from typing import Tuple

import networkx as nx
from anndata import AnnData
from matplotlib import pyplot as plt

from stlearn.utils import _read_graph


def tree_plot_simple(
    adata: AnnData,
    library_id: str | None = None,
    figsize: Tuple[float, float] = (10, 4),
    data_alpha: float = 1.0,
    use_label: str = "louvain",
    spot_size: float | int = 50,
    fontsize: int = 6,
    piesize: float = 0.15,
    zoom: float = 0.1,
    name: str | None = None,
    output: str | None = None,
    dpi: int = 180,
    show_all: bool = False,
    show_plot: bool = True,
    ncols: int = 4,
    copy: bool = False,
) -> AnnData | None:
    """\
    Hierarchical tree plot represent for the global spatial trajectory inference.

    Parameters
    ----------
    adata
        Annotated data matrix.
    library_id
        Library id stored in AnnData.
    use_label
        Use label result of cluster method.
    figsize
        Change figure size.
    data_alpha
        Opacity of the spot.
    fontsize
        Choose font size.
    piesize
        Choose the size of cropped image.
    zoom
        Choose zoom factor.
    show_all
        Show all cropped image or not.
    show_legend
        Show legend or not.
    dpi
        Set dpi as the resolution for the plot.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Nothing
    """

    G = _read_graph(adata, "PTS_graph")

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    G.remove_node(9999)

    start_nodes = []
    disconnected_nodes = []
    for node in G.in_degree():
        if node[1] == 0:
            start_nodes.append(node[0])

    for node in G.out_degree():
        if node[1] == 0:
            disconnected_nodes.append(node[0])

    start_nodes = list(set(start_nodes) - set(disconnected_nodes))
    start_nodes.sort()

    nrows = math.ceil(len(start_nodes) / ncols)

    superfig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.ravel()

    for idx in range(0, nrows * ncols):
        try:
            generate_tree_viz(
                adata, use_label, G, axs[idx], starter_node=start_nodes[idx]
            )
        except:
            axs[idx] = axs[idx].axis("off")

    if name is None:
        name = use_label

    if output is not None:
        superfig.savefig(
            output + "/" + name, dpi=dpi, bbox_inches="tight", pad_inches=0
        )

    if show_plot:
        plt.show()

    return adata


def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G:
        the graph (must be a tree)
    root:
        the root node of current branch
        - if the tree is directed and this is not given,
            the root will be found and used
        - if the tree is directed and this is given, then
            the positions will be just for the descendants of this node.
        - if the tree is undirected and not given,
            then a random choice will be used.
    width:
        horizontal space allocated for this branch - avoids overlap with other branches
    vert_gap:
        gap between levels of hierarchy
    vert_loc:
        vertical location of root
    xcenter:
        horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(G))
            )  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(
        G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None
    ):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def generate_tree_viz(adata, use_label, G, axis, starter_node):
    tmp_edges = []
    for edge in G.edges():
        if starter_node == edge[0]:
            tmp_edges.append(edge)
    tmp_D = nx.DiGraph()
    tmp_D.add_edges_from(tmp_edges)

    pos = hierarchy_pos(tmp_D)
    a = axis

    a.axis("off")
    colors = []
    for n in tmp_D:
        subset = adata.obs[adata.obs["sub_cluster_labels"] == str(n)]
        colors.append(adata.uns[use_label + "_colors"][int(subset[use_label][0])])

    nx.draw_networkx_edges(
        tmp_D,
        pos,
        ax=a,
        arrowstyle="-",
        edge_color="#ADABAF",
        connectionstyle="angle3,angleA=0,angleB=90",
    )
    nx.draw_networkx_nodes(tmp_D, pos, node_color=colors, ax=a)
    nx.draw_networkx_labels(tmp_D, pos, font_color="black", ax=a)
