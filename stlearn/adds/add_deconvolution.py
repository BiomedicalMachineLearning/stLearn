from typing import Optional, Union
from anndata import AnnData
import pandas as pd
import numpy as np
from pathlib import Path


def add_deconvolution(
    adata: AnnData,
    annotation_path: Union[Path, str],
    copy: bool = False,
) -> Optional[AnnData]:

    """\
    Adding label transfered from Seurat

    Parameters
    ----------
    adata
        Annotated data matrix.
    annotation_path
        Path of the output of label transfer result by Seurat
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **[cluster method name]_anno** : `adata.obs` field
        The annotation of cluster results.

    """
    label = pd.read_csv(annotation_path, index_col=0)
    label = label[adata.obs_names]

    adata.obsm["deconvolution"] = label[adata.obs.index].T
