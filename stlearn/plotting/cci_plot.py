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
    use_cluster: str = 'louvain',
    use_het: str = 'het',
    dpi: int = 100,
    spot_size: Union[float,int] = 6.5,
    vmin: int = None,
    vmax: int = None,
    name: str = None,
    output: str = None,
):
    """ Plot tissue clusters and cluster heterogeneity using heatmap
    Parameters
    ----------
    adata: AnnData                  The data object to plot
    use_cluster: str                The clustering results to use
    use_het: str                    Cluster heterogeneity count results from tl.cci.het
    dpi: bool                       Dots per inch
    spot_size: Union[float,int]     Spot size
    
    Returns
    -------
    N/A
    """

    plt.rcParams['figure.dpi'] = dpi
    fig, ax = plt.subplots()
    num_clusters = len(set(adata.obs[use_cluster])) + 1
    ax.set_prop_cycle('color',plt.cm.rainbow(np.linspace(0,1,num_clusters)))
    for item in set(adata.obs[use_cluster]):
        ax.scatter(np.array(adata.obs[adata.obs[use_cluster]==item]['imagecol']), 
                   -np.array(adata.obs[adata.obs[use_cluster]==item]['imagerow']), 
                   alpha=0.6, s=spot_size, edgecolors='none')
    ax.legend(range(num_clusters))
    ax.grid(False)
    plt.axis('equal')

    if name is None:
        name = use_cluster
    if output is not None:
        fig.savefig(output + "/" + name + "_scatter.pdf", dpi=dpi, bbox_inches='tight', pad_inches=0)

    plt.rcParams['figure.dpi'] = dpi * 0.8
    plt.subplots()
    sns.heatmap(adata.uns[use_het], vmin=vmin, vmax=vmax)
    plt.axis('equal')

    if output is not None:
        plt.savefig(output + "/" + name + "_heatmap.pdf", dpi=dpi, bbox_inches='tight', pad_inches=0)

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
        plt.savefig(output + "/" + name + ".pdf", dpi=dpi, bbox_inches='tight', pad_inches=0)

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
        plt.savefig(output + "/" + name + ".pdf", dpi=dpi, bbox_inches='tight', pad_inches=0)

    plt.show()