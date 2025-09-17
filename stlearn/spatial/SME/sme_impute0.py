import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from stlearn.spatial.SME._weighting_matrix import _WEIGHTING_MATRIX, _PLATFORM, \
    weight_matrix, impute_neighbour


def SME_impute0(
    adata: AnnData,
    use_data: str = "raw",
    weights: _WEIGHTING_MATRIX = "weights_matrix_all",
    platform: _PLATFORM = "Visium",
    copy: bool = False,
) -> AnnData | None:
    """\
    Fill missing/zero expression values using spatial, morphological,
    and expression (SME) information when you what to correct for technical noise
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
    """
    adata = adata.copy() if copy else adata

    if use_data == "raw":
        if isinstance(adata.X, csr_matrix):
            count_embed = adata.X.toarray()
        elif isinstance(adata.X, np.ndarray):
            count_embed = adata.X
        elif isinstance(adata.X, pd.Dataframe):
            count_embed = adata.X.values
        else:
            raise ValueError(
                f"""\
                    {type(adata.X)} is not a valid type.
                    """
            )
    else:
        count_embed = adata.obsm[use_data]

    weight_matrix(adata, platform=platform)

    impute_neighbour(adata, count_embed=count_embed, weights=weights)

    imputed_data = adata.obsm["imputed_data"].astype(float)
    mask = count_embed != 0
    count_embed_ = count_embed.astype(float)
    count_embed_[count_embed_ == 0] = np.nan
    adjusted_count_matrix = np.nanmean(np.array([count_embed_, imputed_data]), axis=0)
    adjusted_count_matrix[mask] = count_embed[mask]

    key_added = use_data + "_SME_imputed"
    adata.obsm[key_added] = adjusted_count_matrix

    print("The data adjusted by SME is added to adata.obsm['" + key_added + "']")

    return adata if copy else None
