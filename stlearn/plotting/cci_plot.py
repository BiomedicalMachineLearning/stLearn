from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import sys
from anndata import AnnData
from typing import Optional, Union

from typing import Optional, Union, Mapping  # Special
from typing import Sequence, Iterable  # ABCs
from typing import Tuple  # Classes

import warnings

from .classes import CciPlot
from .classes_bokeh import BokehCciPlot
from ..utils import Empty,_empty,_AxesSubplot

from bokeh.io import push_notebook,output_notebook
from bokeh.plotting import show


def het_plot(
    adata: AnnData,
    # plotting param
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
    cell_alpha: Optional[float] = 1.0,
    use_raw: Optional[bool] = False,
    fname: Optional[str] = None,
    dpi: Optional[int] = 120,
    # cci param
    use_het: Optional[str] = "het",
    contour: bool = False,
    step_size: Optional[int] = None
) -> Optional[AnnData]:


    CciPlot(adata,
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
        use_raw=use_raw,
        fname=fname,
        dpi=dpi,
        use_het=use_het,
        contour=contour,
        step_size=step_size,)


def het_plot_interactive(
    adata: AnnData
    ):
    bokeh_object = BokehCciPlot(adata)
    output_notebook()
    show(bokeh_object.app,notebook_handle=True)



def grid_plot(
    adata: AnnData,
    use_het: str = None,
    num_row: int = 10,
    num_col: int = 10,
    vmin: float = None,
    vmax: float = None,
    cropped: bool = True,
    margin: int = 100,
    dpi: int = 100,
    name: str = None,
    output: str = None,
    copy: bool = False,
) -> Optional[AnnData]:

    """
    Cell diversity plot for sptial transcriptomics data.

    Parameters
    ----------
    adata:                  Annotated data matrix.
    use_het:                Cluster heterogeneity count results from tl.cci.het
    num_row: int            Number of grids on height
    num_col: int            Number of grids on width
    cropped                 crop image or not.
    margin                  margin used in cropping.
    dpi:                    Set dpi as the resolution for the plot.
    name:                   Name of the output figure file.
    output:                 Save the figure as file or not.
    copy:                   Return a copy instead of writing to adata.
    
    Returns
    -------
    Nothing
    """

    plt.subplots()

    sns.heatmap(pd.DataFrame(np.array(adata.uns[use_het]).reshape(num_col, num_row)).T, vmin=vmin, vmax=vmax)
    plt.axis('equal')

    if output is not None:
        plt.savefig(output + "/" + name + "_heatmap.pdf", dpi=dpi, bbox_inches='tight', pad_inches=0)

    plt.show()
