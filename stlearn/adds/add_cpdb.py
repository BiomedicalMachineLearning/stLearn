from anndata import AnnData
import pandas as pd
from pathlib import Path
import os


def cpdb(
    adata: AnnData,
    cpdb_filepath: str = None,
    sep: str = '\t',
    copy: bool = False,
) -> AnnData:
    """ Add CPDB results into AnnData object

    Parameters
    ----------
    adata: AnnData          The data object to add CPDB results into
    cpdb_filepath: str      The path to the CPDB results file
    sep: str                Separator of the CPDB results file
    copy: bool              Copy flag indicating copy or direct edit

    Returns
    -------
    adata: AnnData          The data object that CPDB results added into

    """
    cpdb = pd.read_csv(cpdb_filepath, sep=sep)
    adata.uns["cpdb"] = cpdb
    print("Added cpdb results to the object!")

    return adata if copy else None
