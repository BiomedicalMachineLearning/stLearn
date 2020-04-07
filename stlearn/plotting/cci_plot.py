from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import sys
from anndata import AnnData

"""
"""

def het_plot(
    adata: AnnData,
    het: pd.DataFrame,
    use_cluster: str = 'louvain',
    dpi: int = 100,
    output: str = None,
):
    plt.rcParams['figure.dpi'] = dpi
    fig, ax = plt.subplots()
    num_clusters = len(set(adata.obs[use_cluster])) + 1
    ax.set_prop_cycle('color', plt.cm.gist_rainbow(
        np.linspace(0, 1, num_clusters)))
    for item in set(adata.obs[use_cluster]):
        ax.scatter(np.array(adata.obs[adata.obs[use_cluster] == item]['imagecol']),
                   -np.array(adata.obs[adata.obs[use_cluster]
                                       == item]['imagerow']),
                   alpha=0.6, edgecolors='none')
    ax.legend(range(num_clusters))
    ax.grid(False)
    plt.axis('equal')

    plt.rcParams['figure.dpi'] = dpi * 0.8
    plt.subplots()
    sns.heatmap(het)
    plt.axis('equal')

    plt.show()


def violin_plot(
    adata: AnnData,
    lr: str,
    use_cluster: str = 'louvain',
    dpi: int = 100,
    output: str = None,
):
    try:
        violin = adata.obsm['lr_neighbours'][[lr]]
    except:
        sys.exit('Please run cci counting and clustering first.')
    violin.columns = ['LR_counts']
    violin['cci_cluster'] = adata.obs['lr_neighbours_' + use_cluster]
    plt.rcParams['figure.dpi'] = dpi
    sns.violinplot(x='cci_cluster', y='LR_counts', data=violin, orient='v')
    plt.show()


def stacked_bar_plot(
    adata: AnnData,
    use_annotation: str,
    dpi: int = 100,
    output: str = None,
):
    sns.set()
    try:
        cci = adata.obs['lr_neighbours_louvain']
    except:
        sys.exit('Please run cci counting and clustering first.')
    try:
        label = adata.obs[use_annotation]
    except:
        sys.exit(
            'spot cell type not found in data.obs[' + use_annotation + ']')
    df = pd.DataFrame(0, index=sorted(set(cci)), columns=set(label))
    for spot in cci.index:
        df.loc[cci[spot], label[spot]] += 1

    # From raw value to percentage
    df2 = df.div(df.sum(axis=1), axis=0)
    plt.rcParams['figure.dpi'] = dpi
    df2.plot(kind='bar', stacked='True', legend=False)
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1), ncol=1)
    plt.show()
