from typing import Optional, Union
from anndata import AnnData
import pandas as pd 
import numpy as np 
import stlearn
from pathlib import Path
from natsort import natsorted

def add_loupe_clusters(
    adata: AnnData,
    loupe_path: Union[Path, str],
    key_add: str = "multiplex",
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
    threshold
        Quantile threshold of label
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **[clustering method name]_anno** : `adata.obs` field
        The annotation of cluster results.

    """
    label = pd.read_csv(loupe_path)


    adata.obs[key_add] = pd.Categorical(
        values=np.array(label[key_add]).astype('U'),
        categories=natsorted(label[key_add].unique().astype('U')),
    )
    


