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

    """\
    Perform kmeans clustering for spatial transcriptomics data

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_clusters 
        The number of clusters to form as well as the number of
        centroids to generate.
    use_data
        Use dimensionality reduction result.
    init
        Method for initialization, defaults to 'k-means++'
    max_iter
        Maximum number of iterations of the k-means algorithm for a
        single run.
    tol
        Relative tolerance with regards to inertia to declare convergence.
    precompute_distances
        Precompute distances (faster but takes more memory).
    random_state
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
    copy_x
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True (default), then the original data is
        not modified, ensuring X is C-contiguous.  If False, the original data
        is modified, and put back before the function returns, but small
        numerical differences may be introduced by subtracting and then adding
        the data mean, in this case it will also not ensure that data is
        C-contiguous which may cause a significant slowdown.
    n_jobs
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
    algorithm
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto" chooses
        "elkan" for dense data and "full" for sparse data.
    key_added
        Key add to adata.obs
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Anndata
    """
    
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