from typing import (
    Optional,
    Tuple,  # Special
)

import matplotlib
from anndata import AnnData
from bokeh.io import output_notebook
from bokeh.plotting import show

from stlearn.plotting._docs import doc_cluster_plot, doc_spatial_base_plot
from stlearn.plotting.classes import ClusterPlot
from stlearn.plotting.classes_bokeh import BokehClusterPlot
from stlearn.utils import _docs_params


@_docs_params(spatial_base_plot=doc_spatial_base_plot, cluster_plot=doc_cluster_plot)
def cluster_plot(
    adata: AnnData,
    # plotting param
    title: Optional["str"] = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "default",
    use_label: str | None = None,
    list_clusters: list | None = None,
    ax: matplotlib.axes.Axes | None = None,
    fig: matplotlib.figure.Figure | None = None,
    show_plot: bool = True,
    show_axis: bool = False,
    show_image: bool = True,
    show_color_bar: bool = True,
    zoom_coord: Tuple[float, float, float, float] | None = None,
    crop: bool = True,
    margin: float = 100,
    size: float = 5,
    image_alpha: float = 1.0,
    cell_alpha: float = 1.0,
    fname: str | None = None,
    dpi: int = 120,
    # cluster plot param
    show_subcluster: bool = False,
    show_cluster_labels: bool = False,
    show_trajectories: bool = False,
    reverse: bool = False,
    show_node: bool = False,
    threshold_spots: int = 5,
    text_box_size: float = 5,
    color_bar_size: float = 10,
    bbox_to_anchor: tuple[float, float] | None = (1, 1),
    # trajectory
    trajectory_node_size: int = 10,
    trajectory_alpha: float = 1.0,
    trajectory_width: float = 2.5,
    trajectory_edge_color: str = "#f4efd3",
    trajectory_arrowsize: int = 17,
) -> AnnData | None:
    """\
    Allows the visualization of a cluster results as the discretes values
    of dot points in the Spatial transcriptomics array. We also support to
    visualize the spatial trajectory results


    Parameters
    -------------------------------------
    {spatial_base_plot}
    {cluster_plot}

    Examples
    -------------------------------------
    >>> import stlearn as st
    >>> adata = st.datasets.example_bcba()
    >>> label = "louvain"
    >>> st.pl.cluster_plot(adata, use_label = label, show_trajectories = True)

    """

    assert use_label is not None, "Please select `use_label` parameter"

    ClusterPlot(
        adata,
        title=title,
        figsize=figsize,
        cmap=cmap,
        use_label=use_label,
        list_clusters=list_clusters,
        ax=ax,
        fig=fig,
        show_plot=show_plot,
        show_axis=show_axis,
        show_image=show_image,
        show_color_bar=show_color_bar,
        zoom_coord=zoom_coord,
        crop=crop,
        margin=margin,
        size=size,
        image_alpha=image_alpha,
        cell_alpha=cell_alpha,
        fname=fname,
        dpi=dpi,
        show_subcluster=show_subcluster,
        show_cluster_labels=show_cluster_labels,
        show_trajectories=show_trajectories,
        reverse=reverse,
        show_node=show_node,
        threshold_spots=threshold_spots,
        text_box_size=text_box_size,
        color_bar_size=color_bar_size,
        bbox_to_anchor=bbox_to_anchor,
        trajectory_node_size=trajectory_node_size,
        trajectory_alpha=trajectory_alpha,
        trajectory_width=trajectory_width,
        trajectory_edge_color=trajectory_edge_color,
        trajectory_arrowsize=trajectory_arrowsize,
    )

    return adata


def cluster_plot_interactive(
    adata: AnnData,
):
    bokeh_object = BokehClusterPlot(adata)
    output_notebook()
    show(bokeh_object.app, notebook_handle=True)
