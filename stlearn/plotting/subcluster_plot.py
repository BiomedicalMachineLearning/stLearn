from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import matplotlib
import numpy as np

from typing import Optional, Union, Mapping  # Special
from typing import Sequence, Iterable  # ABCs
from typing import Tuple  # Classes

from anndata import AnnData
import warnings

from stlearn.plotting.classes import SubClusterPlot
from stlearn.plotting._docs import doc_spatial_base_plot, doc_subcluster_plot
from stlearn.utils import _AxesSubplot, Axes, _docs_params


@_docs_params(
    spatial_base_plot=doc_spatial_base_plot, subcluster_plot=doc_subcluster_plot
)
def subcluster_plot(
    adata: AnnData,
    # plotting param
    title: Optional["str"] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = "jet",
    use_label: Optional[str] = None,
    list_clusters: Optional[list] = None,
    ax: Optional[_AxesSubplot] = None,
    show_plot: Optional[bool] = True,
    show_axis: Optional[bool] = False,
    show_image: Optional[bool] = True,
    show_color_bar: Optional[bool] = True,
    crop: Optional[bool] = True,
    margin: Optional[bool] = 100,
    size: Optional[float] = 5,
    image_alpha: Optional[float] = 1.0,
    cell_alpha: Optional[float] = 1.0,
    fname: Optional[str] = None,
    dpi: Optional[int] = 120,
    # subcluster plot param
    cluster: Optional[int] = 0,
    threshold_spots: Optional[int] = 5,
    text_box_size: Optional[float] = 5,
    bbox_to_anchor: Optional[Tuple[float, float]] = (1, 1),
) -> Optional[AnnData]:
    """\
    Allows the visualization of a subclustering results as the discretes values
    of dot points in the Spatial transcriptomics array.

    Parameters
    -------------------------------------
    {spatial_base_plot}
    {subcluster_plot}

    Examples
    -------------------------------------
    >>> import stlearn as st
    >>> adata = st.datasets.example_bcba()
    >>> label = "louvain"
    >>> cluster = 6
    >>> st.pl.cluster_plot(adata, use_label = label, cluster = cluster)

    """

    assert use_label != None, "Please select `use_label` parameter"
    assert (
        use_label in adata.obs.columns
    ), "Please run `stlearn.spatial.cluster.localization` function!"

    SubClusterPlot(
        adata,
        title=title,
        figsize=figsize,
        cmap=cmap,
        use_label=use_label,
        list_clusters=list_clusters,
        ax=ax,
        show_plot=show_plot,
        show_axis=show_axis,
        show_image=show_image,
        show_color_bar=show_color_bar,
        crop=crop,
        margin=margin,
        size=size,
        image_alpha=image_alpha,
        cell_alpha=cell_alpha,
        fname=fname,
        dpi=dpi,
        text_box_size=text_box_size,
        bbox_to_anchor=bbox_to_anchor,
        cluster=cluster,
        threshold_spots=threshold_spots,
    )
