import numpy as np
import scanpy
from anndata import AnnData


def filter_genes(
    adata: AnnData,
    min_counts: int | None = None,
    min_cells: int | None = None,
    max_counts: int | None = None,
    max_cells: int | None = None,
    inplace: bool = True,
) -> AnnData | None | tuple[np.ndarray, np.ndarray]:
    """\
    Wrap function scanpy.pp.filter_genes

    Filter genes based on number of cells or counts.
    Keep genes that have at least `min_counts` counts or are expressed in at
    least `min_cells` cells or have at most `max_counts` counts or are expressed
    in at most `max_cells` cells.
    Only provide one of the optional parameters `min_counts`, `min_cells`,
    `max_counts`, `max_cells` per call.
    Parameters
    ----------
    adata
        An annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    min_counts
        Minimum number of counts required for a gene to pass filtering.
    min_cells
        Minimum number of cells expressed required for a gene to pass filtering.
    max_counts
        Maximum number of counts required for a gene to pass filtering.
    max_cells
        Maximum number of cells expressed required for a gene to pass filtering.
    inplace
        Perform computation inplace or return result.
    Returns
    -------
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix
    gene_subset
        Boolean index mask that does filtering. `True` means that the
        gene is kept. `False` means the gene is removed.
    number_per_gene
        Depending on what was tresholded (`counts` or `cells`), the array stores
        `n_counts` or `n_cells` per gene.
    """

    return scanpy.pp.filter_genes(
        adata,
        min_counts=min_counts,
        min_cells=min_cells,
        max_counts=max_counts,
        max_cells=max_cells,
        inplace=inplace,
    )
