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

def cluster_plot(
    adata: AnnData,
    name: str = None,
    use_label: str = "louvain",
    list_cluster: list = None,
    data_alpha: float = 1.0,
    tissue_alpha: float = 1.0,
    title: str = None,
    spot_size: Union[float,int] = 6.5,
    show_color_bar: bool = True,
    show_axis: bool = False,
    show_legend: bool = True,
    dpi: int = 180,
    output: str = None,
    copy: bool = False,
) -> Optional[AnnData]:
    
    plt.rcParams['figure.dpi'] = dpi


    #tmp = adata.obs
    #tmp[use_data] = tmp[use_data].astype(int)
    n_clusters = len(adata.obs[use_label].unique())

    # Option for turning off showing figure
    plt.ioff()

    # Initialize matplotlib
    fig, a = plt.subplots()

    
    from stlearn.external.scanpy.plotting import palettes
    cmaps = matplotlib.colors.LinearSegmentedColormap.from_list("",palettes.vega_20_scanpy)
    
    cmap_ = plt.cm.get_cmap(cmaps)

    if list_cluster is not None:
        tmp = adata.obs.loc[adata.obs[use_label].isin(np.array(list_cluster).astype(str))]
    else:
        tmp = adata.obs

    # Plot scatter plot based on pixel of spots
    for i,cluster in tmp.groupby(use_label):

        _ = a.scatter(cluster['imagecol'], cluster['imagerow'], c=[cmap_(int(i)/19)], label=int(i),
                      edgecolor="none", alpha=data_alpha,s=spot_size,marker="o")

    if show_legend:
        lgnd = a.legend(bbox_to_anchor=(1.2, 1.0),labelspacing=0.05,fontsize=8,handleheight=1.,edgecolor='white')
        for handle in lgnd.legendHandles:
            handle.set_sizes([20.0])

            

    if not show_axis:
        a.axis('off')

    # Overlay the tissue image
    a.imshow(adata.uns["tissue_img"],alpha=tissue_alpha, zorder=-1,)

    if name is None:
        name = use_label

    if output is not None:
        fig.savefig(output + "/" + name + ".png", dpi=dpi,bbox_inches='tight',pad_inches=0)

    #fig_np = get_img_from_fig(fig,dpi)
    
    plt.show()



