from typing import Optional, Union
from anndata import AnnData
from pathlib import Path
import os
import pandas as pd


def labels(
    adata: AnnData,
    label_filepath: str = None,
    index_col: int = 0,
    sep: str = "\t",
    copy: bool = False,
) -> Optional[AnnData]:
    """Add label transfer results into AnnData object

    Parameters
    ----------
    adata: AnnData          The data object to add L-R info into
    label_filepath: str     The path to the label transfer results file
    sep: str                Separator of the CPDB results file
    copy: bool              Copy flag indicating copy or direct edit

    Returns
    -------
    adata: AnnData          The data object that L-R added into

    """
    labels = pd.read_csv(label_filepath, index_col=index_col, sep=sep)
    adata.uns["label_transfer"] = labels.drop(
        ["predicted.id", "prediction.score.max"], axis=1
    )
    adata.obs["predictions"] = labels["predicted.id"]
    print("label transfer results added to adata.uns['label_transfer']")
    print("predicted label added to adata.obs['predictions'].")

    return adata if copy else None
