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
    use_data: str = None,
    library_id: str = None,
    data_alpha: float = 1.0,
    tissue_alpha: float = 1.0,
    cmap: str = "Spectral_r",
    spot_size: Union[float, int] = 6.5,
    show_color_bar: bool = True,
    show_axis: bool = False,
    cropped: bool = True,
    margin: int = 100,
    dpi: int = 192,
    name: str = None,
    output: str = None,
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Plotting microenvironment.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_data
        Use dimensionality reduction result data.
    library_id
        Library id stored in AnnData.
    data_alpha
        Opacity of the spot.
    tissue_alpha
        Opacity of the tissue.
    cmap
        Color map to use.
    spot_size
        Size of the spot.
    show_color_bar
        Show color bar or not.
    show_axis
        Show axis or not.
    show_legend
        Show legend or not.
    dpi
        Set dpi as the resolution for the plot.
    name
        Name of the output figure file.
    output
        Save the figure as file or not.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Nothing
    """

    colors = _microenv_plot(adata, use_data)

    n_factor = len(colors)
    plt.ioff()

    if "plots" not in adata.uns:
        adata.uns['plots'] = {}

    adata.uns['plots'].update({use_data: {}})

    imagecol = adata.obs["imagecol"]
    imagerow = adata.obs["imagerow"]

    for i in range(0, n_factor):
        fig, a = plt.subplots()
        vmin = min(colors[i])
        vmax = max(colors[i])
        sc = a.scatter(adata.obs["imagecol"], adata.obs["imagerow"], edgecolor="none", alpha=data_alpha, s=spot_size, marker="o",
                       vmin=vmin, vmax=vmax, cmap=plt.get_cmap(cmap), c=colors[i])

        if show_color_bar:
            cb = plt.colorbar(sc, cax=fig.add_axes([0.78, 0.3, 0.03, 0.38]))
            cb.outline.set_visible(False)
        if not show_axis:
            a.axis('off')

        if library_id is None:
            library_id = list(adata.uns["spatial"].keys())[0]

        image = adata.uns["spatial"][library_id]["images"][adata.uns["spatial"]["use_quality"]]
        # Overlay the tissue image
        a.imshow(image, alpha=tissue_alpha, zorder=-1,)


        if cropped:
            a.set_xlim(imagecol.min() - margin,
                    imagecol.max() + margin)

            a.set_ylim(imagerow.min() - margin,
                    imagerow.max() + margin)
            
            a.set_ylim(a.get_ylim()[::-1])
            #plt.gca().invert_yaxis()

        if name is None:
            name = method
        if output is not None:
            fig.savefig(output + "/factor_" + str(i+1) + ".png",
                        dpi=dpi, bbox_inches='tight', pad_inches=0)

        fig_np = get_img_from_fig(fig, dpi)

        plt.close(fig)

        current_plot = {"factor_"+str(i+1): fig_np}

        adata.uns['plots'][use_data].update(current_plot)

    print("The plot stored in adata.uns['plots']['" + use_data + "']")


def _microenv_plot(adata, use_data):

    n_factor = adata.obsm[use_data].shape[1]
    l_colors = []
    for i in range(0, n_factor):
        colors = adata.obsm[use_data][:, i]
        vmin = min(colors)
        vmax = max(colors)
        l_colors.append(colors)

    return l_colors
