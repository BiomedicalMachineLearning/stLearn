from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import pandas as pd
from anndata import AnnData
from typing import Optional, Union
import numpy as np
from natsort import natsorted

def clustgeo(
    adata: AnnData,
    n_clusters: int = 12,
    use_data: str = "X_pca",
    alpha: float = 0.5,
    key_added: str = "clustgeo",
    copy: bool = False,
) -> Optional[AnnData]:

    
    print("Applying Ward-like hierarchical clustering ...")
        
    stats = importr("stats")
    clustgeo = importr("ClustGeo")
    pandas2ri.activate()

    attributes = pd.DataFrame(adata.obsm[use_data])
    coordinates = pandas2ri.py2ri(adata.obs[["imagerow","imagecol"]])
    D0 = stats.dist(attributes)
    D1 = stats.dist(coordinates)
    tree = clustgeo.hclustgeo(D0,D1,alpha=alpha)


    def dollar(obj, name):
        """R's "$"."""
        return obj[obj.do_slot('names').index(name)]


    label = stats.cutree(tree,n_clusters)


    adata.obs[key_added] = pd.Categorical(
        values=np.array(label).astype('U'),
        categories=natsorted(np.unique(np.array(label)).astype('U')),
    )
    
    print('Ward-like hierarchical clustering is done! The labels are stored in adata.obs["clustgeo"]')
    
    return adata if copy else None
    