from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import matplotlib
import numpy as np

from stlearn._compat import Literal
from typing import Optional, Union
from anndata import AnnData
import warnings
#from .utils import get_img_from_fig, checkType


def non_spatial_plot(
    adata: AnnData,
    use_label: str = "louvain",
    dpi: int = 180
) -> Optional[AnnData]:
    
    """\
    A wrap function to plot all the non-spatial plot from scanpy.

    This function will produce 3 plots: PAGA graph, clustering plot in PAGA space and
    DPT in PAGA space.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_label
        Use label result of clustering method.
    dpi
        Set dpi as the resolution for the plot.
    Returns
    -------
    Nothing
    """

    plt.rcParams['figure.dpi'] = dpi

    if 'paga' in adata.uns.keys():
        adata.uns[use_label+"_colors"] = adata.uns["tmp_color"]
        from stlearn.external.scanpy.api.pl import paga
        print("PAGA plot:")

        paga(adata, color=use_label)

        from stlearn.external.scanpy.api.tl import draw_graph

        draw_graph(adata, init_pos='paga')
        adata.uns[use_label+"_colors"] = adata.uns["tmp_color"]

        from stlearn.external.scanpy.api.pl import draw_graph

        print("Gene expression (reduced dimension) plot:")
        draw_graph(adata, color=use_label, legend_loc='on data')

        print("Diffusion pseudotime plot:")
        draw_graph(adata, color="dpt_pseudotime")

    else:
        from stlearn.external.scanpy.api.tl import draw_graph

        draw_graph(adata)
        adata.uns[use_label+"_colors"] = adata.uns["tmp_color"]

        from stlearn.external.scanpy.api.pl import draw_graph
        draw_graph(adata, color=use_label, legend_loc='on data')
