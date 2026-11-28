from typing import Any

import numpy as np
import scanpy
from anndata import AnnData
from numpy.random.mtrand import RandomState
from scipy.sparse import spmatrix


def run_pca(
    data: AnnData | np.ndarray | spmatrix,
    n_comps: int = 50,
    *,
    svd_solver: str = "arpack",
    random_state: int | RandomState | None = 0,
    **kwargs: Any,
) -> AnnData | np.ndarray | spmatrix | tuple | None:
    """\
    Wrap function scanpy.pp.pca

    Principal component analysis [Pedregosa11]_.
    Computes PCA coordinates, loadings and variance decomposition.
    Uses the implementation of *scikit-learn* [Pedregosa11]_.

    Only the highly variable genes in `.var['highly_variable']` are used when
    they have been determined beforehand, otherwise all genes are used. Pass
    `mask=None` to use all genes regardless.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to genes.
    n_comps
        Number of principal components to compute.
    svd_solver
        SVD solver to use:

        - `'arpack'` (the default - deterministic) for the ARPACK wrapper in
          SciPy (:func:`~scipy.sparse.linalg.svds`)
        - `'randomized'` for the randomized algorithm due to Halko (2009).
        - `'auto'` chooses automatically depending on the size of the problem.

    random_state
        Change to use different initial states for the optimization.
    **kwargs
        Any other keyword argument is passed to :func:`scanpy.pp.pca`, for
        example `mask`, `zero_center`, `dtype`, `chunked`, `chunk_size` or
        `return_info`.

    Returns
    -------
    X_pca: :class:`~scipy.sparse.spmatrix`, :class:`~numpy.ndarray`
        If `data` is array-like and `return_info=False` was passed,
        this function only returns `X_pca`.
    None
        If `data` is an :class:`~anndata.AnnData`, it adds these fields to it:

        - `.obsm['X_pca']` - PCA representation of data.
        - `.varm['PCs']` - The principal components containing the loadings.
        - `.uns['pca']['variance_ratio']` - Ratio of explained variance.
        - `.uns['pca']['variance']` - Explained variance, equivalent to the
          eigenvalues of the covariance matrix.

    """

    result = scanpy.pp.pca(
        data,
        n_comps=n_comps,
        svd_solver=svd_solver,
        random_state=random_state,
        **kwargs,
    )

    print(
        "PCA is done! Generated in adata.obsm['X_pca'], adata.uns['pca'] and "
        + "adata.varm['PCs']"
    )

    return result
