"""
Plotting of continuous features stored in adata.obs.
"""

from typing import (
    Optional,  # Special
)

import matplotlib
from anndata import AnnData

from stlearn.plotting.classes import FeaturePlot


# @_docs_params(spatial_base_plot=doc_spatial_base_plot, gene_plot=doc_gene_plot)
def feat_plot(
    adata: AnnData,
    feature: str = None,
    threshold: float | None = None,
    contour: bool = False,
    step_size: int | None = None,
    title: Optional["str"] = None,
    figsize: tuple[float, float] | None = None,
    cmap: str | None = "Spectral_r",
    use_label: str | None = None,
    list_clusters: list | None = None,
    ax: matplotlib.axes.Axes | None = None,
    fig: matplotlib.figure.Figure | None = None,
    show_plot: bool | None = True,
    show_axis: bool | None = False,
    show_image: bool | None = True,
    show_color_bar: bool | None = True,
    color_bar_label: str | None = "",
    zoom_coord: float | None = None,
    crop: bool | None = True,
    margin: float | None = 100,
    size: float | None = 7,
    image_alpha: float | None = 1.0,
    cell_alpha: float | None = 0.7,
    use_raw: bool | None = False,
    fname: str | None = None,
    dpi: int | None = 120,
    vmin: float | None = None,
    vmax: float | None = None,
) -> AnnData | None:
    """\
    Allows the visualization of a continuous features stored in adata.obs
     for Spatial transcriptomics array.


    Parameters
    -------------------------------------
    {spatial_base_plot}
    {feature_plot}

    Examples
    -------------------------------------
    >>> import stlearn as st
    >>> adata = st.datasets.example_bcba()
    >>> st.pl.gene_plot(adata, 'dpt_pseudotime')

    """
    FeaturePlot(
        adata,
        feature=feature,
        threshold=threshold,
        contour=contour,
        step_size=step_size,
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
        color_bar_label=color_bar_label,
        zoom_coord=zoom_coord,
        crop=crop,
        margin=margin,
        size=size,
        image_alpha=image_alpha,
        cell_alpha=cell_alpha,
        use_raw=use_raw,
        fname=fname,
        dpi=dpi,
        vmin=vmin,
        vmax=vmax,
    )
