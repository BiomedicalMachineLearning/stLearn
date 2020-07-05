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

    annotation = []
    for i in np.sort(data.obs[use_label].unique().astype(int)):
        c_index = data.obs[data.obs[use_label] == str(i)].index
        tmp = label[c_index].sum(axis=1)
        tmp = tmp/np.sum(tmp)
        annotation.append(", ".join(tmp[tmp > np.quantile(tmp,threshold)].index + " " + np.round(np.array(tmp[tmp > np.quantile(tmp,threshold)])*100,2).astype(str) + "%"))
        
    if len(annotation) > len(set(annotation)):
        duplicated = list(set([x for x in annotation if annotation.count(x) > 1]))
        for name in duplicated:
            indices = [i for i, x in enumerate(annotation) if x == name]
            for i in range(0,len(indices)):
                annotation[indices[i]] = annotation[indices[i]] + " " + str(i+1)

    stlearn.add.annotation(data,label_list=annotation,
                 use_label=use_label)


