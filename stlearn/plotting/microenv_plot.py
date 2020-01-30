from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import matplotlib
import numpy as np

from stlearn._compat import Literal
from typing import Optional, Union
from anndata import AnnData
import warnings
from .utils import get_img_from_fig

def microenv_plot(
    adata: AnnData,
    name: str = None,
    use_data: str = None,
    genes: Optional[Union[str,list]] = None,
    data_alpha: float = 1.0,
    tissue_alpha: float = 1.0,
    cmap: str = "Spectral_r",
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    spot_size: Union[float,int] = 6.5,
    show_color_bar: bool = True,
    show_axis: bool = False,
    dpi: int = 192,
    output: str = None,
    copy: bool = False,
) -> Optional[AnnData]:
    
    colors = _microenv_plot(adata,use_data)

    n_factor = len(colors)
    plt.ioff()

    if "plots" not in adata.uns:
        adata.uns['plots'] = {}
            

    adata.uns['plots'].update({use_data: {}})

    for i in range(0,n_factor):
        fig, a = plt.subplots()
        vmin = min(colors[i])
        vmax = max(colors[i])
        sc = a.scatter(adata.obs["imagecol"], adata.obs["imagerow"], edgecolor="none", alpha=data_alpha,s=spot_size,marker="o",
               vmin=vmin, vmax=vmax,cmap=plt.get_cmap(cmap),c=colors[i])

        if show_color_bar:
            cb = plt.colorbar(sc,cax = fig.add_axes([0.78, 0.3, 0.03, 0.38]))
            cb.outline.set_visible(False)
        if not show_axis:
            a.axis('off')

        # Overlay the tissue image
        a.imshow(adata.uns["tissue_img"],alpha=tissue_alpha, zorder=-1,)

        if output is not None:
            fig.savefig(output + "/factor_" + str(i+1) + ".png", dpi=dpi,bbox_inches='tight',pad_inches=0)

            
        fig_np = get_img_from_fig(fig,dpi)

        plt.close(fig)

        current_plot = {"factor_"+str(i+1):fig_np}

        adata.uns['plots'][use_data].update(current_plot) 

    print("The plot stored in adata.uns['plots']['" + use_data + "']")





def _microenv_plot(adata,use_data):


    n_factor = adata.obsm[use_data].shape[1]
    l_colors = []
    for i in range(0,n_factor):
        colors = adata.obsm[use_data][:,i]
        vmin = min(colors)
        vmax = max(colors)
        l_colors.append(colors)

    return l_colors