from pathlib import Path

import pandas as pd
from anndata import AnnData


def add_deconvolution(
    adata: AnnData,
    annotation_path: Path | str,
    copy: bool = False,
) -> AnnData | None:
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
