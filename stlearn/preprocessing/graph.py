
from types import MappingProxyType
from typing import Union, Optional, Any, Mapping, Callable

import numpy as np
import scipy
from anndata import AnnData
from numpy.random import RandomState
from .._compat import Literal
import scanpy

_Method = Literal['umap', 'gauss', 'rapids']
_MetricFn = Callable[[np.ndarray, np.ndarray], float]
# from sklearn.metrics.pairwise_distances.__doc__:
_MetricSparseCapable = Literal[
    'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'
]
_MetricScipySpatial = Literal[
    'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming',
    'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto',
    'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
    'yule'
]
_Metric = Union[_MetricSparseCapable, _MetricScipySpatial]


def neighbors(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None,
    use_rep: Optional[str] = None,
    knn: bool = True,
    random_state: Optional[Union[int, RandomState]] = 0,
    method: Optional[_Method] = 'umap',
    metric: Union[_Metric, _MetricFn] = 'euclidean',
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Compute a neighborhood graph of observations [McInnes18]_.
    The neighbor search efficiency of this heavily relies on UMAP [McInnes18]_,
    which also provides a method for estimating connectivities of data points -
    the connectivity of the manifold (`method=='umap'`). If `method=='gauss'`,
    connectivities are computed according to [Coifman05]_, in the adaption of
    [Haghverdi16]_.
    Parameters
    ----------
    adata
        Annotated data matrix.
    n_neighbors
        The size of local neighborhood (in terms of number of neighboring data
        points) used for manifold approximation. Larger values result in more
        global views of the manifold, while smaller values result in more local
        data being preserved. In general values should be in the range 2 to 100.
        If `knn` is `True`, number of nearest neighbors to be searched. If `knn`
        is `False`, a Gaussian kernel width is set to the distance of the
        `n_neighbors` neighbor.
    {n_pcs}
    {use_rep}
    knn
        If `True`, use a hard threshold to restrict the number of neighbors to
        `n_neighbors`, that is, consider a knn graph. Otherwise, use a Gaussian
        Kernel to assign low weights to neighbors more distant than the
        `n_neighbors` nearest neighbor.
    random_state
        A numpy random seed.
    method
        Use 'umap' [McInnes18]_ or 'gauss' (Gauss kernel following [Coifman05]_
        with adaptive width [Haghverdi16]_) for computing connectivities.
        Use 'rapids' for the RAPIDS implementation of UMAP (experimental, GPU
        only).
    metric
        A known metric’s name or a callable that returns a distance.
    metric_kwds
        Options for the metric.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, updates or returns `adata` with the following:
    **connectivities** : sparse matrix (`.uns['neighbors']`, dtype `float32`)
        Weighted adjacency matrix of the neighborhood graph of data
        points. Weights should be interpreted as connectivities.
    **distances** : sparse matrix (`.uns['neighbors']`, dtype `float32`)
        Instead of decaying weights, this stores distances for each pair of
        neighbors.
    """

    scanpy.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep,
              knn=knn, random_state=random_state, method=method, metric=metric,
              metric_kwds=metric_kwds, copy=copy)

    print("Created k-Nearest-Neighbor graph in adata.uns['neighbors'] ")
