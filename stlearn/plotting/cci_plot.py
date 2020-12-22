from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import sys
from anndata import AnnData
from typing import Optional, Union


def het_plot(
    adata: AnnData,
    use_het: str = 'het',
    library_id: str = None,
    data_alpha: float = 1.0,
    tissue_alpha: float = 1.0,
    vmin: float = None,
    vmax: float = None,
    cmap: str = "Spectral_r",
    spot_size: Union[float, int] = 6.5,
    show_legend: bool = False,
    show_color_bar: bool = True,
    show_axis: bool = False,
    cropped: bool = True,
    margin: int = 100,
    dpi: int = 100,
    name: str = None,
    output: str = None,
    copy: bool = False,
) -> Optional[AnnData]:

    """
    Cell type diversity plot for sptial transcriptomics data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_het:
        Cluster heterogeneity count results from tl.cci.het
    library_id
        Library id stored in AnnData.
    data_alpha
        Opacity of the spot.
    tissue_alpha
        Opacity of the tissue.
    vmin
        Scalar of the plot.
    vmax
        Scalar of the plot.
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
    cropped
        crop image or not.
    margin
        margin used in cropping.
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

    colors = adata.uns[use_het].tolist()

    # Option for turning off showing figure
    plt.ioff()

    # Initialize matplotlib
    fig, a = plt.subplots()

    if not vmin:
        vmin = min(colors)
    if not vmax:
        vmax = max(colors)
    # Plot scatter plot based on pixel of spots
    plot = a.scatter(adata.obs["imagecol"], adata.obs["imagerow"], edgecolor="none", alpha=data_alpha, s=spot_size, marker="o",
                     vmin=vmin, vmax=vmax, cmap=plt.get_cmap(cmap), c=colors)

    if show_color_bar:

        cb = plt.colorbar(plot, cax=fig.add_axes(
            [0.78, 0.3, 0.03, 0.38]), cmap=cmap)
        cb.outline.set_visible(False)

    if not show_axis:
        a.axis('off')

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"][adata.uns["spatial"]["use_quality"]]
    # Overlay the tissue image
    a.imshow(image, alpha=tissue_alpha, zorder=-1,)

    imagecol = adata.obs["imagecol"]
    imagerow = adata.obs["imagerow"]

    if cropped:
        a.set_xlim(imagecol.min() - margin, imagecol.max() + margin)

        a.set_ylim(imagerow.min() - margin, imagerow.max() + margin)

        a.set_ylim(a.get_ylim()[::-1])

    if output is not None:
        fig.savefig(output + "/" + name + ".png", dpi=dpi,
                    bbox_inches='tight', pad_inches=0)
    
    plt.show()

    return


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
