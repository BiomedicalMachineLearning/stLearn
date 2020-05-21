import logging as logg
from typing import Union, Optional, Tuple, Collection, Sequence, Iterable
from anndata import AnnData
import numpy as np
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix
from numpy.random.mtrand import RandomState
import scanpy


def run_pca(
    data: Union[AnnData, np.ndarray, spmatrix],
    n_comps: int = 50,
    zero_center: Optional[bool] = True,
    svd_solver: str = 'auto',
    random_state: Optional[Union[int, RandomState]] = 0,
    return_info: bool = False,
    use_highly_variable: Optional[bool] = None,
    dtype: str = 'float32',
    copy: bool = False,
    chunked: bool = False,
    chunk_size: Optional[int] = None,
) -> Union[AnnData, np.ndarray, spmatrix]:
    """\
    Wrap function scanpy.pp.pca
    Principal component analysis [Pedregosa11]_.
    Computes PCA coordinates, loadings and variance decomposition.
    Uses the implementation of *scikit-learn* [Pedregosa11]_.
    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    n_comps
        Number of principal components to compute.
    zero_center
        If `True`, compute standard PCA from covariance matrix.
        If `False`, omit zero-centering variables
        (uses :class:`~sklearn.decomposition.TruncatedSVD`),
        which allows to handle sparse input efficiently.
        Passing `None` decides automatically based on sparseness of the data.
    svd_solver
        SVD solver to use:
        `'arpack'`
          for the ARPACK wrapper in SciPy (:func:`~scipy.sparse.linalg.svds`)
        `'randomized'`
          for the randomized algorithm due to Halko (2009).
        `'auto'` (the default)
          chooses automatically depending on the size of the problem.
    random_state
        Change to use different initial states for the optimization.
    return_info
        Only relevant when not passing an :class:`~anndata.AnnData`:
        see “**Returns**”.
    use_highly_variable
        Whether to use highly variable genes only, stored in
        `.var['highly_variable']`.
        By default uses them if they have been determined beforehand.
    dtype
        Numpy data type string to which to convert the result.
    copy
        If an :class:`~anndata.AnnData` is passed, determines whether a copy
        is returned. Is ignored otherwise.
    chunked
        If `True`, perform an incremental PCA on segments of `chunk_size`.
        The incremental PCA automatically zero centers and ignores settings of
        `random_seed` and `svd_solver`. If `False`, perform a full PCA.
    chunk_size
        Number of observations to include in each chunk.
        Required if `chunked=True` was passed.
    Returns
    -------
    X_pca : :class:`~scipy.sparse.spmatrix`, :class:`~numpy.ndarray`
        If `data` is array-like and `return_info=False` was passed,
        this function only returns `X_pca`…
    adata : anndata.AnnData
        …otherwise if `copy=True` it returns or else adds fields to `adata`:
        `.obsm['X_pca']`
             PCA representation of data.
        `.varm['PCs']`
             The principal components containing the loadings.
        `.uns['pca']['variance_ratio']`
             Ratio of explained variance.
        `.uns['pca']['variance']`
             Explained variance, equivalent to the eigenvalues of the
             covariance matrix.
    """

    scanpy.pp.pca(data, n_comps=n_comps, zero_center=zero_center, svd_solver=svd_solver,
            random_state=random_state, return_info=return_info,
            use_highly_variable=use_highly_variable,
            dtype=dtype, copy=copy, chunked=chunked, chunk_size=chunk_size)

    print("PCA is done! Generated in adata.obsm['X_pca'], adata.uns['pca'] and adata.varm['PCs']")