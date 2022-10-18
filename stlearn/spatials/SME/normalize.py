from typing import Optional
from anndata import AnnData
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from ._weighting_matrix import (
    calculate_weight_matrix,
    impute_neighbour,
    _WEIGHTING_MATRIX,
    _PLATFORM,
)


def SME_normalize(
    adata: AnnData,
    use_data: str = "raw",
    weights: _WEIGHTING_MATRIX = "weights_matrix_all",
    platform: _PLATFORM = "Visium",
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    using spatial location (S), tissue morphological feature (M) and gene expression (E) information to normalize data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_data
        Input data, can be `raw` counts or log transformed data
    weights
        Weighting matrix for imputation.
        if `weights_matrix_all`, matrix combined all information from spatial location (S),
        tissue morphological feature (M) and gene expression (E)
        if `weights_matrix_pd_md`, matrix combined information from spatial location (S),
        tissue morphological feature (M)
    platform
        `Visium` or `Old_ST`
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Anndata
    """
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

    calculate_weight_matrix(adata, platform=platform)

    impute_neighbour(adata, count_embed=count_embed, weights=weights)

    imputed_data = adata.obsm["imputed_data"].astype(float)
    imputed_data[imputed_data == 0] = np.nan
    adjusted_count_matrix = np.nanmean(np.array([count_embed, imputed_data]), axis=0)

    key_added = use_data + "_SME_normalized"
    adata.obsm[key_added] = adjusted_count_matrix

    print("The data adjusted by SME is added to adata.obsm['" + key_added + "']")

    return adata if copy else None
