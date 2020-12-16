import numpy as np
import pandas as pd
from anndata import AnnData


def merge(
    adata: AnnData,
    use_lr: str = "cci_lr",
    use_het: str = "cci_het",
) -> AnnData:
    """Merge results from cell type heterogeneity and L-R clustering
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count
    use_lr: str             CCI LR scores
    use_het: str            CCI HET scores

    Returns
    -------
    adata: AnnData          With merged result stored in adata.uns['merged']
    """

    adata.uns["merged"] = adata.uns[use_het].mul(adata.uns[use_lr])

    print(
        "Results of spatial interaction analysis has been written to adata.uns['merged']"
    )

    return adata
