from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import matplotlib
import numpy as np

from stlearn._compat import Literal
from typing import Optional, Union
from anndata import AnnData
import warnings


# from .utils import get_img_from_fig, checkType


def gene_plot(
        adata: AnnData,
        method: str = "CumSum",
        genes: Optional[Union[str, list]] = None,
        list_clusters: list = None,
        use_raw_count: bool = False,
        use_label: str = "louvain",
        threshold: float = None,
        library_id: str = None,
        data_alpha: float = 1.0,
        tissue_alpha: float = 1.0,
        cmap: str = "Spectral_r",
        spot_size: Union[float, int] = 6.5,
        show_legend: bool = False,
        show_color_bar: bool = True,
        show_axis: bool = False,
        cropped: bool = True,
        margin: int = 100,
        name: str = None,
        dpi: int = 150,
        vmin: float = None,
        vmax: float = None,
        output: str = None,
        copy: bool = False,
) -> Optional[AnnData]:
    """\
    Gene expression plot for sptial transcriptomics data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    library_id
        Library id stored in AnnData.
    method
        Use method to count. We prorive: NaiveMean, NaiveSum, CumSum.
    genes
        Choose a gene or a list of genes.
    threshold
        Threshold to filter genes
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
    show_trajectory
        Show the spatial trajectory or not. It requires stlearn.spatial.trajectory.pseudotimespace.
    show_subcluster
        Show subcluster or not. It requires stlearn.spatial.trajectory.global_level.
    name
        Name of the output figure file.
    dpi
        DPI of the output figure.
    vmin
        minimum value for color.
    vmax
        maximum value for color.
    output
        Save the figure as file or not.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Nothing
    """

    # plt.rcParams['figure.dpi'] = dpi

    def create_query(list_clusters, use_label):
        ini = ""
        for sub in list_clusters:
            ini = ini + use_label + ' == "' + str(sub) + '" | '
        return ini[:-2]

    if list_clusters is not None:
        query_adata = adata[
            adata.obs.query(create_query(list_clusters, use_label)).index
        ].copy()
    else:
        query_adata = adata.copy()

    if use_raw_count:
        query_adata.X = query_adata.layers["raw_count"]

    if type(genes) == str:
        genes = [genes]
    colors = _gene_plot(query_adata, method, genes)

    if threshold is not None:
        colors = colors[colors > threshold]

    index_filter = colors.index

    filter_obs = query_adata.obs.loc[index_filter]

    imagecol = filter_obs["imagecol"]
    imagerow = filter_obs["imagerow"]

    # Option for turning off showing figure
    plt.ioff()

    # Initialize matplotlib
    fig, a = plt.subplots()

    if not vmin:
        vmin = min(colors)
    if not vmax:
        vmax = max(colors)
    # Plot scatter plot based on pixel of spots
    plot = a.scatter(
        imagecol,
        imagerow,
        edgecolor="none",
        alpha=data_alpha,
        s=spot_size,
        marker="o",
        vmin=vmin,
        vmax=vmax,
        cmap=plt.get_cmap(cmap),
        c=colors,
    )

    if show_color_bar:
        cb = plt.colorbar(plot, aspect=10, shrink=0.5, cmap=cmap)
        cb.outline.set_visible(False)

    if not show_axis:
        a.axis("off")

    if library_id is None:
        library_id = list(query_adata.uns["spatial"].keys())[0]

    image = query_adata.uns["spatial"][library_id]["images"][
        query_adata.uns["spatial"]["use_quality"]
    ]
    # Overlay the tissue image
    a.imshow(
        image,
        alpha=tissue_alpha,
        zorder=-1,
    )

    if cropped:
        imagecol = query_adata.obs["imagecol"]
        imagerow = query_adata.obs["imagerow"]

        a.set_xlim(imagecol.min() - margin, imagecol.max() + margin)

        a.set_ylim(imagerow.min() - margin, imagerow.max() + margin)

        a.set_ylim(a.get_ylim()[::-1])

    if name is None:
        name = method
    if output is not None:
        fig.savefig(output + "/" + name, dpi=dpi, bbox_inches="tight", pad_inches=0)

    plt.show()


def _gene_plot(adata, method, genes):
    # Gene plot option

    if len(genes) == 0:
        raise ValueError("Genes shoule be provided, please input genes")

    elif len(genes) == 1:

        if genes[0] not in adata.var.index:
            raise ValueError(
                genes[0] + " is not exist in the data, please try another gene"
            )

        colors = adata[:, genes].to_df().iloc[:, -1]

        return colors
    else:

        for gene in genes:
            if gene not in adata.var.index:
                genes.remove(gene)
                warnings.warn(
                    "We removed " + gene + " because they not exist in the data"
                )

            if len(genes) == 0:
                raise ValueError("All provided genes are not exist in the data")

        count_gene = adata[:, genes].to_df()

        if method is None:
            raise ValueError(
                "Please provide method to combine genes by NaiveMean/NaiveSum/CumSum"
            )

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
