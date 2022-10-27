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

from stlearn.plotting.classes import GenePlot
from stlearn.plotting.classes_bokeh import BokehGenePlot
from stlearn.plotting._docs import doc_spatial_base_plot, doc_gene_plot
from stlearn.utils import Empty, _empty, _AxesSubplot, _docs_params

from bokeh.io import push_notebook, output_notebook
from bokeh.plotting import show


@_docs_params(spatial_base_plot=doc_spatial_base_plot, gene_plot=doc_gene_plot)
def gene_plot(
    adata: AnnData,
    gene_symbols: Union[str, list] = None,
    threshold: Optional[float] = None,
    method: str = "CumSum",
    contour: bool = False,
    step_size: Optional[int] = None,
    title: Optional["str"] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = "Spectral_r",
    use_label: Optional[str] = None,
    list_clusters: Optional[list] = None,
    ax: Optional[matplotlib.axes._subplots.Axes] = None,
    fig: Optional[matplotlib.figure.Figure] = None,
    show_plot: Optional[bool] = True,
    show_axis: Optional[bool] = False,
    show_image: Optional[bool] = True,
    show_color_bar: Optional[bool] = True,
    color_bar_label: Optional[str] = "",
    zoom_coord: Optional[float] = None,
    crop: Optional[bool] = True,
    margin: Optional[bool] = 100,
    size: Optional[float] = 7,
    image_alpha: Optional[float] = 1.0,
    cell_alpha: Optional[float] = 0.7,
    use_raw: Optional[bool] = False,
    fname: Optional[str] = None,
    dpi: Optional[int] = 120,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Optional[AnnData]:
    """\
    Allows the visualization of a single gene or multiple genes as the values
    of dot points or contour in the Spatial transcriptomics array.


    Parameters
    -------------------------------------
    {spatial_base_plot}
    {gene_plot}

    Examples
    -------------------------------------
    >>> import stlearn as st
    >>> adata = st.datasets.example_bcba()
    >>> genes = ["BRCA1","BRCA2"]
    >>> st.pl.gene_plot(adata, gene_symbols = genes)

    """
    GenePlot(
        adata,
        gene_symbols=gene_symbols,
        threshold=threshold,
        method=method,
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


def gene_plot_interactive(adata: AnnData):
    bokeh_object = BokehGenePlot(adata)
    output_notebook()
    show(bokeh_object.app, notebook_handle=True)
