from anndata import AnnData
import pandas as pd
from pathlib import Path
import os

def cpdb(
    adata: AnnData,
    cpdb_filepath: str = None,
    sep: str = '\t',
) -> AnnData:

    cpdb = pd.read_csv(cpdb_filepath, sep=sep)
    adata.uns["cpdb"] = cpdb
    print("Added cpdb results to the object!")

    return adata