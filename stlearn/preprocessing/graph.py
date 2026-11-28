from typing import Any

import scanpy
from anndata import AnnData
from numpy.random import RandomState


def neighbors(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    *,
    random_state: int | RandomState | None = 0,
    **kwargs: Any,
) -> AnnData | None:
    """\
    Compute a neighborhood graph of observations [McInnes18]_.
    The neighbor search efficiency of this heavily relies on UMAP [McInnes18]_,
    which also provides a method for estimating connectivities of data points -
    the connectivity of the manifold (`method=='umap'`). If `method=='gauss'`,
    connectivities are computed according to [Coifman05]_, in the adaption of
    [Haghverdi16]_.

    Parameters
    ----------
    adata:
        Annotated data matrix.
    n_neighbors:
        The size of local neighborhood (in terms of number of neighboring data
        points) used for manifold approximation. Larger values result in more
        global views of the manifold, while smaller values result in more local
        data being preserved. In general values should be in the range 2 to 100.
        If `knn` is `True`, number of nearest neighbors to be searched. If `knn`
        is `False`, a Gaussian kernel width is set to the distance of the
        `n_neighbors` neighbor.
    n_pcs:
        Use this many PCs. If `n_pcs==0` use `.X` if `use` is `None`.
    random_state:
        A numpy random seed.
    **kwargs:
        Any other keyword argument is passed to :func:`scanpy.pp.neighbors`, for
        example `use` (such as `use="obsm.X_pca"`), `knn`, `method`, `metric`
        or `metric_kwds`.

    Returns
    -------
    Updates `adata` with the following:
    **connectivities** : sparse matrix (`.uns['neighbors']`, dtype `float32`)
        Weighted adjacency matrix of the neighborhood graph of data
        points. Weights should be interpreted as connectivities.
    **distances** : sparse matrix (`.uns['neighbors']`, dtype `float32`)
        Instead of decaying weights, this stores distances for each pair of
        neighbors.
    """

    adata = scanpy.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        random_state=random_state,
        **kwargs,
    )

    print("Created k-Nearest-Neighbor graph in adata.uns['neighbors'] ")

    return adata
