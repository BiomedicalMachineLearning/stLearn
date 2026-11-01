from anndata import AnnData

from ._adjust import adjust_by_blocks, average_with_imputed, expression
from ._weighting_matrix import (
    _PLATFORM,
    _WEIGHTING_MATRIX,
    neighbour_weights,
)


def sme_normalize(
    adata: AnnData,
    use_data: str = "raw",
    weights: _WEIGHTING_MATRIX = "weights_matrix_all",
    platform: _PLATFORM = "Visium",
    copy: bool = False,
) -> AnnData | None:
    """\
    Reduce technical noise by spatially smoothing all expression values using
     spatial, morphological, and expression (SME) information.

    This function modified ALL expression values by averaging each spot's expression
    with weighted contributions from similar neighbors. It modifies ALL expression
    values to reduce technical noise across the entire dataset.

    Each spot is adjusted by its 3 most similar neighbours (or 6 for physical distance).
    No spots x spots matrix is created - only possible neighbours. Weightings without
    physical distance still compare every pair of spots, so they are slow on large
    samples.

    Parameters
    ----------
    adata:
        Annotated data matrix.
    use_data:
        Input data, can be `raw` counts or log transformed data
    weights : _WEIGHTING_MATRIX, default="weights_matrix_all"
        Strategy for computing neighbor similarity weights:
        - "weights_matrix_all": Combines spatial location (S) +
        morphological features (M) + gene expression correlation (E).
        - "weights_matrix_pd_gd": Physical distance + gene expression correlation only.
        - "weights_matrix_pd_md": Physical distance + morphological features only.
        - "weights_matrix_gd_md": Gene expression + morphological features only.
        - "gene_expression_correlation": Expression similarity only.
        - "physical_distance": Spatial proximity only.
        - "morphological_distance": Tissue morphology similarity only.
    platform:
        `Visium` or `Old_ST`
    copy:
        If True, return a copy instead of writing to adata. If False, modify adata
        in place and return None.
    Returns
    -------
    AnnData or None
        Adds `adata.obsm[use_data + "_SME_normalized"]`, sparse when the data
        is sparse.
    """
    adata = adata.copy() if copy else adata

    data = expression(adata, use_data)
    neighbours, top_weights = neighbour_weights(
        adata, weights=weights, platform=platform
    )
    adata.obsm["top_weights"] = top_weights

    key_added = use_data + "_SME_normalized"
    adata.obsm[key_added] = adjust_by_blocks(average_with_imputed, data, neighbours)

    print("The data adjusted by SME is added to adata.obsm['" + key_added + "']")

    return adata if copy else None
