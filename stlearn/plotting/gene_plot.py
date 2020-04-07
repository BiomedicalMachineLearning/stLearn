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


def gene_plot(
    adata: AnnData,
    name: str = None,
    method: str = None,
    genes: Optional[Union[str, list]] = None,
    data_alpha: float = 1.0,
    tissue_alpha: float = 1.0,
    cmap: str = "Spectral_r",
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    spot_size: Union[float, int] = 6.5,
    show_legend: bool = False,
    show_color_bar: bool = True,
    show_axis: bool = False,
    dpi: int = 192,
    output: str = None,
    copy: bool = False,
) -> Optional[AnnData]:

    plt.rcParams['figure.dpi'] = dpi

    if type(genes) == str:
        genes = [genes]
    colors = _gene_plot(adata, method, genes)

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

    # Overlay the tissue image
    a.imshow(adata.uns["tissue_img"], alpha=tissue_alpha, zorder=-1,)

    if name is None:
        name = method
    if output is not None:
        fig.savefig(output + "/" + name + ".png", dpi=dpi,
                    bbox_inches='tight', pad_inches=0)

    plt.show()
    # Store figure
    #fig_np = get_img_from_fig(fig,dpi)
    # plt.close(fig)
    # if "plots" not in adata.uns:
    #    adata.uns['plots'] = {}
    #adata.uns['plots'].update({str(', '.join(genes)):fig_np})
    # print("The plot stored in adata.uns['plot']['" +
    #    str(', '.join(genes)) + "']")


def _gene_plot(adata, method, genes):

    # Gene plot option

    if len(genes) == 0:
        raise ValueError('Genes shoule be provided, please input genes')

    elif len(genes) == 1:

        if genes[0] not in adata.var.index:
            raise ValueError(
                genes[0] + ' is not exist in the data, please try another gene')

        colors = list(adata[:, genes].X.toarray().reshape(1, -1)[0])

        return colors
    else:

        for gene in genes:
            if gene not in adata.var.index:
                genes.remove(gene)
                warnings.warn("We removed " + gene +
                              " because they not exist in the data")

            if len(genes) == 0:
                raise ValueError(
                    'All provided genes are not exist in the data')

        count_gene = adata[:, genes].to_df()

        if method is None:
            raise ValueError(
                'Please provide method to combine genes by NaiveMean/NaiveSum/CumSum')

        if method == "NaiveMean":
            present_genes = (count_gene > 0).sum(axis=1) / len(genes)

            count_gene = (count_gene.mean(axis=1)) * present_genes
        elif method == "NaiveSum":
            present_genes = (count_gene > 0).sum(axis=1) / len(genes)

            count_gene = (count_gene.sum(axis=1)) * present_genes

        elif method == "CumSum":
            count_gene = count_gene.cumsum(axis=1).iloc[:, -1]

        colors = count_gene

        return colors
