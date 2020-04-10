from typing import Optional, Union
from anndata import AnnData
from pathlib import Path
import os
import pandas as pd


def lr(
    adata: AnnData,
    copy: bool = False,
) -> Optional[AnnData]:
    """ Add significant Ligand-Receptor pairs into AnnData object

    Parameters
    ----------
    adata: AnnData          The data object to add L-R info into
    copy: bool              Copy flag indicating copy or direct edit

    Returns
    -------
    adata: AnnData          The data object that L-R added into

    """
    lr = adata.uns['cpdb']['interacting_pair'].to_list()
    lr2 = [i for i in lr if 'complex' not in i]
    lr3 = [i for i in lr2 if ' ' not in i]
    lr4 = [i for i in lr3 if i.count('_') == 1]
    adata.uns['lr'] = lr4
    print("Added ligand receptor pairs to adata.uns['lr'].")

    return adata if copy else None
