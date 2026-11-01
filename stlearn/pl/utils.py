import io
from collections.abc import Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory, offset_copy
from PIL import Image
from scanpy.plotting import palettes

from stlearn.pl import palettes_st
from stlearn.utils import _cluster_positions

# Named palettes: lists of distinct colours, used in order.
PALETTES = {
    "vega_10_scanpy": palettes.vega_10_scanpy,
    "vega_20_scanpy": palettes.vega_20_scanpy,
    "default_102": palettes.default_102,
    "default_28": palettes.default_28,
    "jana_40": palettes_st.jana_40,
    "default": palettes_st.default,
}


def figure_and_axes(fig=None, ax=None, **subplots_kwargs):
    """The figure and axes to draw on, and whether they are new.

    `ax` if given, on `fig` or else its own figure; `fig`'s current axes if only
    `fig` is given; otherwise a new figure and axes from `plt.subplots`.
    """
    if ax is not None:
        return (ax.figure if fig is None else fig), ax, False
    if fig is not None:
        return fig, fig.gca(), False
    fig, ax = plt.subplots(**subplots_kwargs)
    return fig, ax, True


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    from io import BytesIO

    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    buf.seek(0)
    img_arr = numpy.frombuffer(buf.getvalue(), dtype=numpy.uint8)
    img = numpy.asarray(Image.open(BytesIO(img_arr)))
    buf.close()
    # img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    return img


def centroidpython(x, y):
    length_of_x = len(x)
    return sum(x) / length_of_x, sum(y) / length_of_x


def subclusters_of(
    adata: AnnData,
    use_label: str,
    clusters: str | Iterable[str],
) -> list[str]:
    """Subclusters of the given cluster labels, from adata.uns["split_node"]."""
    split_node = adata.uns["split_node"]
    return [
        subcluster
        for position in _cluster_positions(adata, use_label, clusters)
        for subcluster in split_node[str(position)]
    ]


def draw_cluster_nodes(
    ax: Axes,
    adata: AnnData,
    use_label: str,
    clusters: str | Iterable[str],
    colors: Sequence[str],
    size: float,
    alpha: float = 1.0,
) -> list[int]:
    """Draw a solid node at the centroid of each subcluster of `clusters`.

    A node takes its cluster's colour: `colors` is indexed by the cluster's
    position in the categories, like adata.uns[f"{use_label}_colors"]. Returns
    the positions of the clusters that have a node.
    """
    centroids = adata.uns["centroid_dict"]
    split_node = adata.uns["split_node"]
    xs, ys, node_colors, drawn = [], [], [], []
    for position in _cluster_positions(adata, use_label, clusters):
        subclusters = split_node[str(position)]
        if len(subclusters) > 0:
            drawn.append(position)
        for subcluster in subclusters:
            x, y = centroids[subcluster]
            xs.append(x)
            ys.append(y)
            node_colors.append(colors[position])
    ax.scatter(
        xs,
        ys,
        s=size,
        c=node_colors,
        edgecolors="black",
        linewidths=1,
        alpha=alpha,
        zorder=100,
    )
    return drawn


def add_cluster_legend(
    ax: Axes,
    labels: Sequence[str],
    colors: Sequence[str],
    fontsize: float = 10,
    bbox_to_anchor: tuple[float, float] | None = (1, 1),
) -> Legend:
    """Legend of cluster labels - by default just outside the right edge of `ax`."""
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="black",
            label=label,
        )
        for label, color in zip(labels, colors, strict=True)
    ]
    return ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=bbox_to_anchor,
        fontsize=fontsize,
        frameon=False,
    )


def place_legend_right_of_colour_bar(legend, ax, colour_bar_ax):
    """Put `legend` past `colour_bar_ax`'s tick labels, level with the top of `ax`."""
    fig = ax.figure
    tight = colour_bar_ax.get_tightbbox()
    labels_width = 0.0 if tight is None else tight.x1 - colour_bar_ax.bbox.x1
    beside = offset_copy(
        colour_bar_ax.transAxes, fig=fig, x=labels_width * 72 / fig.dpi, units="points"
    )
    legend.set_bbox_to_anchor(
        (1, 1), transform=blended_transform_factory(beside, ax.transAxes)
    )


def check_sublist(full, sub):
    set_sub = set(sub)
    index_bool = [x in set_sub for x in full]

    return index_bool


def get_cmap(cmap):
    """Checks inputted cmap string."""
    if isinstance(cmap, str) and cmap in PALETTES:
        cmap = PALETTES[cmap]
    # If refers to matplotlib cmap
    elif isinstance(cmap, str):
        cmap_n = plt.get_cmap(cmap).N
        return plt.get_cmap(cmap), cmap_n
    elif isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):  # already cmap
        cmap_n = cmap.N
        return cmap, cmap_n

    cmap_n = len(cmap)
    cmaps = matplotlib.colors.LinearSegmentedColormap.from_list("", cmap)
    cmap_ = matplotlib.colormaps.get_cmap(cmaps)
    return cmap_, cmap_n


def check_cmap(cmap):
    """Initialize cmap"""
    scanpy_cmap = ["vega_10_scanpy", "vega_20_scanpy", "default_102", "default_28"]
    stlearn_cmap = ["jana_40", "default"]
    cmap_available = plt.colormaps() + scanpy_cmap + stlearn_cmap
    error_msg = (
        "cmap must be a matplotlib.colors.LinearSegmentedColormap OR"
        "one of these: " + str(cmap_available)
    )
    if isinstance(cmap, str):
        assert cmap in cmap_available, error_msg
    elif not isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):
        raise Exception(error_msg)

    return cmap


def get_colors(adata, obs_key, cmap="default", label_set=None):
    """Retrieves colors if present in adata.uns, if not present then will set
    them as per scanpy & return in order requested. If fewer colors are present
    than there are categories, the existing colors are kept and the remainder
    are generated.
    """
    col_key = f"{obs_key}_colors"

    if not hasattr(adata.obs[obs_key], "cat"):  # Ensure categorical
        adata.obs[obs_key] = adata.obs[obs_key].astype("category")
    labels_ordered = adata.obs[obs_key].cat.categories

    colors_ordered = list(adata.uns.get(col_key, []))
    n_labels = len(labels_ordered)
    if len(colors_ordered) < n_labels:
        check_cmap(cmap)
        palette = PALETTES.get(cmap) if isinstance(cmap, str) else None
        if palette is not None and n_labels <= len(palette):
            # A palette's own colours, in order. Sampling across it instead would
            # blend neighbouring colours into washed-out ones.
            new_colors = [matplotlib.colors.to_hex(color) for color in palette]
        else:
            # A colormap, or a palette with fewer colours than clusters: spread
            # the clusters evenly across it.
            cmap_, _ = get_cmap(cmap)
            new_colors = [
                matplotlib.colors.rgb2hex(cmap_(i / max(n_labels - 1, 1)))
                for i in range(n_labels)
            ]
        colors_ordered += new_colors[len(colors_ordered) : n_labels]
        adata.uns[col_key] = colors_ordered

    # Returning the colors of the desired labels in indicated order #
    if label_set is not None:
        colors_ordered = [
            colors_ordered[numpy.where(labels_ordered == label)[0][0]]
            for label in label_set
        ]

    return colors_ordered
