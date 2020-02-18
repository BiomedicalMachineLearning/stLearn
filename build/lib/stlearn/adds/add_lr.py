from typing import Optional, Union
from anndata import AnnData
from pathlib import Path
import os


def lr(
    adata: AnnData,
    lr_path: Union[Path, str],
    sep: str = "\t",
    column: str = "interacting_pair",
    copy: bool = False,
) -> Optional[AnnData]:
    if lr_path is not None and os.path.isfile(lr_path):
        try:
            import pandas as pd
            lr = pd.read_csv(lr_path, sep=sep)[column].to_list()
            lr2 = [i for i in lr if 'complex' not in i]
            lr3 = [i for i in lr2 if ' ' not in i]
            lr4 = [i for i in lr3 if i.count('_') == 1]
            adata.uns["lr_means"] = lr4
            print("Added ligand recepter file to the object!")
            
            return adata if copy else None
        except:
            raise ValueError(f'''\
            {lr_path!r} does not end on a valid extension.
            ''')
    else:
        raise ValueError(f'''\
        {lr_path!r} does not end on a valid extension.
        ''')
    return adata if copy else None