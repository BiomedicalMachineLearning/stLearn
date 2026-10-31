from anndata import AnnData

from stlearn.spatial.sme._adjust import adjust_by_blocks, expression, fill_zeros
from stlearn.spatial.sme._weighting_matrix import (
    _PLATFORM,
    _WEIGHTING_MATRIX,
    neighbour_weights,
)


def sme_impute0(
    adata: AnnData,
    use_data: str = "raw",
    weights: _WEIGHTING_MATRIX = "weights_matrix_all",
    platform: _PLATFORM = "Visium",
    copy: bool = False,
) -> AnnData | None:
    """\
    Fill missing/zero expression values using spatial, morphological,
    and expression (sme) information when you what to correct for technical noise
    (dropouts) without altering existing biological signals.

    This function replaces only zero/missing values with spatially-informed
    predictions while preserving all original non-zero expression measurements.

    Parameters
    ----------
    adata :
        Annotated data matrix must contain obsm["X_morphology"] and obsm["X_pca"].
    use_data :
        input data, can be `raw` counts or log transformed data
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
    platform :
        `Visium` or `Old_ST`
    copy :
        If True, return a copy instead of writing to adata. If False, modify adata
        in place and return None.
    Returns
    -------
    AnnData or None
        Adds `adata.obsm[use_data + "_SME_imputed"]`, sparse when the data is
        sparse.
    """
    adata = adata.copy() if copy else adata

    data = expression(adata, use_data)
    neighbours, top_weights = neighbour_weights(
        adata, weights=weights, platform=platform
    )
    adata.obsm["top_weights"] = top_weights

    key_added = use_data + "_SME_imputed"
    adata.obsm[key_added] = adjust_by_blocks(fill_zeros, data, neighbours)

    print("The data adjusted by sme is added to adata.obsm['" + key_added + "']")

    return adata if copy else None
