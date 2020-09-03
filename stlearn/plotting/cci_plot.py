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
    cmap: str = "Spectral_r",
    spot_size: Union[float, int] = 6.5,
    show_legend: bool = False,
    show_color_bar: bool = True,
    show_axis: bool = False,
    dpi: int = 192,
    spot_size: Union[float,int] = 6.5,
    vmin: int = None,
    vmax: int = None,
    name: str = None,
    output: str = None,
    copy: bool = False,
) -> Optional[AnnData]:

    """
    Cell diversity plot for sptial transcriptomics data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_het:
        Cluster heterogeneity count results from tl.cci.het
    library_id
        Library id stored in AnnData.
    method
        Use method to count. We prorive: NaiveMean, NaiveSum, CumSum.
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
    show_trajectory
        Show the spatial trajectory or not. It requires stlearn.spatial.trajectory.pseudotimespace.
    show_subcluster
        Show subcluster or not. It requires stlearn.spatial.trajectory.global_level.
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

    plt.rcParams['figure.dpi'] = dpi

    colors = adata.uns[use_het].tolist()

    # Option for turning off showing figure
    plt.ioff()

    # Initialize matplotlib
    fig, a = plt.subplots()

    vmin = min(colors)
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

    if name is None:
        name = method
    if output is not None:
        fig.savefig(output + "/" + name, dpi=plt.figure().dpi,
                    bbox_inches='tight', pad_inches=0)

    plt.show()


def violin_plot(
    adata: AnnData,
    lr: str,
    use_cluster: str = 'louvain',
    dpi: int = 100,
    name: str = None,
    output: str = None,
):
    """ Plot the distribution of CCI counts within spots of each CCI clusters
    Parameters
    ----------
    adata: AnnData          The data object to plot
    lr: str                 The specified Ligand-Receptor pair to plot
    use_cluster: str        The clustering results to use
    dpi: bool               Dots per inch
    name: str               Save as file name
    output: str             Save to directory
    Returns
    -------
    N/A
    """
    try:
        violin = adata.obsm['lr_neighbours'][[lr]]
    except:
        sys.exit('Please run cci counting and clustering first.')
    violin.columns = ['LR_counts']
    violin['cci_cluster'] = adata.obs['lr_neighbours_' + use_cluster]
    plt.rcParams['figure.dpi'] = dpi
    sns.violinplot(x='cci_cluster', y='LR_counts', data=violin, orient='v')
    if name is None:
        name = use_cluster

    if output is not None:
        plt.savefig(output + "/" + name, dpi=plt.figure().dpi, bbox_inches='tight', pad_inches=0)
    plt.show()
    

def stacked_bar_plot(
    adata: AnnData,
    use_annotation: str,
    dpi: int = 100,
    name: str = None,
    output: str = None,
):
    """ Plot the proportion of cell types in each CCI cluster
    Parameters
    ----------
    adata: AnnData          The data object to plot
    use_annotation: str     The cell type annotation to be used in plotting
    dpi: bool               Dots per inch
    name: str               Save as file name
    output: str             Save to directory
    Returns
    -------
    N/A
    """
    sns.set()
    try:
        cci = adata.obs['lr_neighbours_louvain']
    except:
        sys.exit('Please run cci counting and clustering first.')
    try:
        label = adata.obs[use_annotation]
    except:
        sys.exit('spot cell type not found in data.obs[' + use_annotation + ']')
    df = pd.DataFrame(0, index=sorted(set(cci)), columns=set(label))
    for spot in cci.index:
        df.loc[cci[spot], label[spot]] += 1

    # From raw value to percentage
    df2 = df.div(df.sum(axis=1), axis=0)
    plt.rcParams['figure.dpi'] = dpi
    df2.plot(kind='bar', stacked='True', legend=False)
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1), ncol=1)
    if name is None:
        name = use_annotation

    if output is not None:
        plt.savefig(output + "/" + name, dpi=plt.figure().dpi, bbox_inches='tight', pad_inches=0)

    plt.show()