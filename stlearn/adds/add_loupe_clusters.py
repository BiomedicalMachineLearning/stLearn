from pathlib import Path

import pandas as pd
from anndata import AnnData
from natsort import natsorted

from stlearn.adds._barcodes import by_barcode


def add_loupe_clusters(
    adata: AnnData,
    loupe_path: Path | str,
    key_add: str = "multiplex",
    copy: bool = False,
) -> AnnData | None:
    """\
    Add loupe clusters to adata.

    Parameters
    ----------
    adata
        Annotated data matrix.
    loupe_path
        Path of the Loupe Browser CSV file: barcodes in the first column.
    key_add
        Column of the CSV with the clusters, and the adata.obs column they go in.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **[key_add]** : `adata.obs` field
        The clusters, matched to spots by barcode; NaN for spots not in the file.

    """
    adata = adata.copy() if copy else adata

    label = by_barcode(pd.read_csv(loupe_path, index_col=0), adata, loupe_path)
    clusters = label[key_add].map(str, na_action="ignore")
    adata.obs[key_add] = pd.Categorical(
        values=clusters.to_numpy(),
        categories=natsorted(clusters.dropna().unique()),
    )

    return adata if copy else None
