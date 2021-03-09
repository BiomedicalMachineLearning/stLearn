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

from .classes import GenePlot
from .classes_bokeh import BokehBasePlot
from ..utils import Empty,_empty,_AxesSubplot

from bokeh.io import push_notebook,output_notebook
from bokeh.plotting import show

def gene_plot(
    adata: AnnData,
    gene_symbols: Union[str, list] = None,
    threshold: Optional[float] = None,
    method: str = "CumSum",
    contour: bool = False,
    step_size: Optional[int] = None,
    title: Optional['str'] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = "Spectral_r",
    use_label: Optional[str] = None,
    list_clusters: Optional[list] = None,
    ax: Optional[_AxesSubplot] = None,
    show_plot: Optional[bool] = True,
    show_axis: Optional[bool] = False,
    show_image: Optional[bool] = True,
    show_color_bar: Optional[bool] = True,
    crop: Optional[bool] = True,
    margin: Optional[bool] = 100,
    size: Optional[float] = 7,
    image_alpha: Optional[float] = 1.0,
    cell_alpha: Optional[float] = 0.7,
    use_raw: Optional[bool] = False,

) -> Optional[AnnData]:

    GenePlot(adata,
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
        show_plot=show_plot,
        show_axis=show_axis,
        show_image=show_image,
        show_color_bar=show_color_bar,
        crop=crop,
        margin=margin,
        size=size,
        image_alpha=image_alpha,
        cell_alpha=cell_alpha,
        use_raw=use_raw
        )

def gene_plot_interactive(
    adata: AnnData
    ):
    bokeh_object = BokehBasePlot(adata)
    output_notebook()
    show(bokeh_object.app,notebook_handle=True)