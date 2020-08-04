from typing import Optional, Union
from anndata import AnnData
import pandas as pd 
import numpy as np 
import stlearn
from pathlib import Path

def auto_annotate(
    adata: AnnData,
    annotation_path: Union[Path, str],
    use_label: str = "louvain",
    threshold: float = 0.9,
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
    use_label
        Choosing clustering type.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **[clustering method name]_anno** : `adata.obs` field
        The annotation of cluster results.

    """
    label = pd.read_csv(annotation_path,index_col=0)
    
    adata.obsm["deconvolution"] = label[adata.obs.index].T