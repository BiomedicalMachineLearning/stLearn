from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import matplotlib
import numpy as np
import networkx as nx

from typing import Optional, Union, Mapping  # Special
from typing import Sequence, Iterable  # ABCs
from typing import Tuple  # Classes

from anndata import AnnData
import warnings

from stlearn.plotting.classes import ClusterPlot
from stlearn.plotting.classes_bokeh import BokehClusterPlot
from stlearn.plotting._docs import doc_spatial_base_plot, doc_cluster_plot
from stlearn.utils import _AxesSubplot, Axes, _docs_params

from bokeh.io import push_notebook, output_notebook
from bokeh.plotting import show


@_docs_params(spatial_base_plot=doc_spatial_base_plot, cluster_plot=doc_cluster_plot)
def cluster_plot(
    adata: AnnData,
    # plotting param
    title: Optional["str"] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = "default",
    use_label: Optional[str] = None,
    list_clusters: Optional[list] = None,
    ax: Optional[matplotlib.axes._subplots.Axes] = None,
    fig: Optional[matplotlib.figure.Figure] = None,
    show_plot: Optional[bool] = True,
    show_axis: Optional[bool] = False,
    show_image: Optional[bool] = True,
    show_color_bar: Optional[bool] = True,
    zoom_coord: Optional[float] = None,
    crop: Optional[bool] = True,
    margin: Optional[bool] = 100,
    size: Optional[float] = 5,
    image_alpha: Optional[float] = 1.0,
    cell_alpha: Optional[float] = 1.0,
    fname: Optional[str] = None,
    dpi: Optional[int] = 120,
    # cluster plot param
    show_subcluster: Optional[bool] = False,
    show_cluster_labels: Optional[bool] = False,
    show_trajectories: Optional[bool] = False,
    reverse: Optional[bool] = False,
    show_node: Optional[bool] = False,
    threshold_spots: Optional[int] = 5,
    text_box_size: Optional[float] = 5,
    color_bar_size: Optional[float] = 10,
    bbox_to_anchor: Optional[Tuple[float, float]] = (1, 1),
    # trajectory
    trajectory_node_size: Optional[int] = 10,
    trajectory_alpha: Optional[float] = 1.0,
    trajectory_width: Optional[float] = 2.5,
    trajectory_edge_color: Optional[str] = "#f4efd3",
    trajectory_arrowsize: Optional[int] = 17,
) -> Optional[AnnData]:

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

    assert use_label != None, "Please select `use_label` parameter"

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


def cluster_plot_interactive(
    adata: AnnData,
):

    bokeh_object = BokehClusterPlot(adata)
    output_notebook()
    show(bokeh_object.app, notebook_handle=True)
