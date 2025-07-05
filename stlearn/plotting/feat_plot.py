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
    feature: str | None = None,
    threshold: float | None = None,
    contour: bool = False,
    step_size: int | None = None,
    title: Optional["str"] = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "Spectral_r",
    use_label: str | None = None,
    list_clusters: list | None = None,
    ax: matplotlib.axes.Axes | None = None,
    fig: matplotlib.figure.Figure | None = None,
    show_plot: bool = True,
    show_axis: bool = False,
    show_image: bool = True,
    show_color_bar: bool = True,
    color_bar_label: str = "",
    zoom_coord: tuple[float, float, float, float] | None = None,
    crop: bool = True,
    margin: float = 100,
    size: float = 7,
    image_alpha: float = 1.0,
    cell_alpha: float = 0.7,
    use_raw: bool = False,
    fname: str | None = None,
    dpi: int = 120,
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
    >>> adata = st.datasets.visium_sge(sample_id="V1_Breast_Cancer_Block_A_Section_1")
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

    return adata
