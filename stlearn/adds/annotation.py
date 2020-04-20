from typing import Optional, Union
from anndata import AnnData
from matplotlib import pyplot as plt
from pathlib import Path
import os


def annotation(
    adata: AnnData,
    label_list: list,
    use_label: str = "louvain",
    copy: bool = False,
) -> Optional[AnnData]:

    """\
    Adding annotation for cluster

    Parameters
    ----------
    adata
        Annotated data matrix.
    label_list
        List of the labels which assigned to current clustering result.
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

    if label_list is None:
        raise ValueError("Please give the label list!")

    if len(label_list) != len(adata.obs[use_label].unique()):
        raise ValueError("Please give the correct number of label list!")

    adata.obs[use_label + '_anno'] = adata.obs[use_label]
    adata.obs[use_label + '_anno'].cat.categories = label_list

    print("The annotation is added to adata.obs['" + use_label + "_anno"+"']")

    return adata if copy else None
