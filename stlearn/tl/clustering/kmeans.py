import numpy as np
import pandas as pd
from anndata import AnnData
from natsort import natsorted
from sklearn.cluster import KMeans


def kmeans(
    adata: AnnData,
    n_clusters: int = 20,
    use_data: str = "X_pca",
    init: str = "k-means++",
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 0.0001,
    random_state: int | np.random.RandomState | None = None,
    copy_x: bool = True,
    algorithm: str = "lloyd",
    key_added: str = "kmeans",
    copy: bool = False,
) -> AnnData | None:
    """\
    Perform kmeans cluster for spatial transcriptomics data

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    n_clusters: int, default = 20
        The number of clusters to form as well as the number of
        centroids to generate.
    use_data: str, default = "X_pca"
        Use dimensionality reduction result.
    init: str, default = "k-means++"
        Method for initialization, defaults to 'k-means++'.
    n_init: int, default = 10
        Number of time the k-means algorithm will be run with different
        centroid seeds.
    max_iter: int, default = 300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    tol: float, default = 0.0001
        Relative tolerance with regard to inertia to declare convergence.
    random_state: int | np.random.RandomState | None, default = None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
    copy_x: bool, default = True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True (default), then the original data is
        not modified, ensuring X is C-contiguous.  If False, the original data
        is modified, and put back before the function returns, but small
        numerical differences may be introduced by subtracting and then adding
        the data mean, in this case it will also not ensure that data is
        C-contiguous which may cause a significant slowdown.
    algorithm: str, default = "lloyd"
        K-means algorithm to use. The classical EM-style algorithm is "lloyd".
        The "elkan" variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality.
    key_added: str, default = "kmeans"
        Key add to adata.obs
    copy: bool, default = False
        Return a copy instead of writing to adata.
    Returns
    -------
    Anndata
    """

    data = adata.obsm[use_data]

    print("Applying Kmeans cluster ...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        copy_x=copy_x,
        algorithm=algorithm,
    ).fit(data)

    adata.obs[key_added] = pd.Categorical(
        values=np.array(kmeans.labels_).astype("U"),
        categories=natsorted(np.unique(np.array(kmeans.labels_)).astype("U")),
    )

    print('Kmeans cluster is done! The labels are stored in adata.obs["kmeans"]')

    return adata if copy else None
