from typing import Literal

import numpy
import pandas
import scanpy
from anndata import AnnData


def highly_variable_genes(
    adata: AnnData,
    *,
    layer: str | None = None,
    n_top_genes: int | None = None,
    min_disp: float = 0.5,
    max_disp: float = numpy.inf,
    min_mean: float = 0.0125,
    max_mean: float = 3,
    span: float = 0.3,
    n_bins: int = 20,
    flavor: Literal["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"] = "seurat",
    subset: bool = False,
    inplace: bool = True,
    batch_key: str | None = None,
    check_values: bool = True,
) -> pandas.DataFrame | None:
    """\
    Wrap function of scanpy.pp.highly_variable_genes
    Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab

    Annotate highly variable genes.
    Expects logarithmized data, except when `flavor='seurat_v3'` or
    `'seurat_v3_paper'`, which expect count data (pass `layer="counts"`) and
    need the `scikit-misc` package.

    Parameters
    ----------
    adata:
        The annotated data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to genes.
    layer:
        If provided, use `adata.layers[layer]` instead of `adata.X`.
    n_top_genes:
        Number of highly variable genes to keep. Mandatory if
        `flavor='seurat_v3'`. When given, the mean and dispersion cutoffs
        below are ignored.
    min_disp, max_disp:
        Cutoffs for the normalized dispersion. Ignored if `n_top_genes` is
        given or `flavor='seurat_v3'`.
    min_mean, max_mean:
        Cutoffs for the mean expression. Ignored if `n_top_genes` is given or
        `flavor='seurat_v3'`.
    span:
        Fraction of the cells used to estimate the variance in the loess model
        fit if `flavor='seurat_v3'`.
    n_bins:
        Number of bins for the mean gene expression. Dispersions are
        normalized within each bin.
    flavor:
        Method for identifying highly variable genes: `'seurat'`,
        `'cell_ranger'`, `'seurat_v3'` or `'seurat_v3_paper'`.
    subset:
        Subset `adata` to the highly variable genes instead of only marking them.
    inplace:
        Whether to store the results in `adata.var` or return them.
    batch_key:
        If given, select highly variable genes within each batch separately
        and merge them.
    check_values:
        Check that counts are integers. Only used for the `seurat_v3` flavors.

    Returns
    -------
    A :class:`pandas.DataFrame` of the computed metrics if `inplace=False`,
    otherwise updates `adata.var` with `highly_variable`, `means` and either
    `dispersions` and `dispersions_norm` or `variances` and `variances_norm`.
    """

    result = scanpy.pp.highly_variable_genes(
        adata,
        layer=layer,
        n_top_genes=n_top_genes,
        min_disp=min_disp,
        max_disp=max_disp,
        min_mean=min_mean,
        max_mean=max_mean,
        span=span,
        n_bins=n_bins,
        flavor=flavor,
        subset=subset,
        inplace=inplace,
        batch_key=batch_key,
        check_values=check_values,
    )
    if inplace:
        n_hvg = int(adata.var["highly_variable"].sum())
        print(
            f"{n_hvg} highly variable genes are marked in adata.var['highly_variable']"
        )
    return result
