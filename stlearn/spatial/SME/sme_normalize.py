import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from ._weighting_matrix import (
    _PLATFORM,
    _WEIGHTING_MATRIX,
    impute_neighbour,
    weight_matrix,
)


def SME_normalize(
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
    imputed_data[imputed_data == 0] = np.nan
    adjusted_count_matrix = np.nanmean(np.array([count_embed, imputed_data]), axis=0)

    key_added = use_data + "_SME_normalized"
    adata.obsm[key_added] = adjusted_count_matrix

    print("The data adjusted by SME is added to adata.obsm['" + key_added + "']")

    return adata if copy else None
