from typing import (
    Optional,  # Special
)

from anndata import AnnData

from stlearn.plotting._docs import doc_spatial_base_plot, doc_subcluster_plot
from stlearn.plotting.classes import SubClusterPlot
from stlearn.utils import _AxesSubplot, _docs_params


@_docs_params(
    spatial_base_plot=doc_spatial_base_plot, subcluster_plot=doc_subcluster_plot
)
def subcluster_plot(
    adata: AnnData,
    # plotting param
    title: Optional["str"] = None,
    figsize: tuple[float, float] | None = None,
    cmap: str | None = "jet",
    use_label: str | None = None,
    list_clusters: list | None = None,
    ax: _AxesSubplot | None = None,
    show_plot: bool | None = True,
    show_axis: bool | None = False,
    show_image: bool | None = True,
    show_color_bar: bool | None = True,
    crop: bool | None = True,
    margin: float | None = 100,
    size: float | None = 5,
    image_alpha: float | None = 1.0,
    cell_alpha: float | None = 1.0,
    fname: str | None = None,
    dpi: int | None = 120,
    # subcluster plot param
    cluster: int | None = 0,
    threshold_spots: int | None = 5,
    text_box_size: float | None = 5,
    bbox_to_anchor: tuple[float, float] | None = (1, 1),
) -> AnnData | None:
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

    assert use_label is not None, "Please select `use_label` parameter"
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
