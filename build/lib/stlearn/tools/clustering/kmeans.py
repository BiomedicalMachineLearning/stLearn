from sklearn.cluster import KMeans
from anndata import AnnData
from typing import Optional, Union
import pandas as pd
import numpy as np
from natsort import natsorted

def kmeans(
    adata: AnnData,
    n_clusters: int = 20,
    use_data: str = "X_pca",
    init: str = "k-means++",
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 0.0001,
    precompute_distances: str = "auto",
    random_state: str = None,
    copy_x: bool = True,
    n_jobs: int = None, 
    algorithm: str = "auto",
    key_added: str = "kmeans",
    copy: bool = False,
) -> Optional[AnnData]:
    
    data = adata.obsm[use_data]
    
    print("Applying Kmeans clustering ...")
        
    kmeans = KMeans(n_clusters=n_clusters,
                    init = init, 
                    n_init=n_init,
                    max_iter=max_iter,
                    tol=tol, 
                    precompute_distances=precompute_distances,
                    random_state=random_state,
                    copy_x=copy_x,
                    n_jobs=n_jobs,
                    algorithm=algorithm).fit(data)

    adata.obs[key_added] = pd.Categorical(
        values=np.array(kmeans.labels_).astype('U'),
        categories=natsorted(np.unique(np.array(kmeans.labels_)).astype('U')),
    )

    print('Kmeans clustering is done! The labels are stored in adata.obs["kmeans"]')
    
    return adata if copy else None