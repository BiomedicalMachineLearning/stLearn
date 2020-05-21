from typing import Optional, Union
import numpy as np
from anndata import AnnData
from numpy.random.mtrand import RandomState
from scipy.sparse import issparse
import scanpy

def run_diffmap(adata: AnnData, n_comps: int = 15, copy: bool = False):
    """\
    Diffusion Maps [Coifman05]_ [Haghverdi15]_ [Wolf18]_.
    Diffusion maps [Coifman05]_ has been proposed for visualizing single-cell
    data by [Haghverdi15]_. The tool uses the adapted Gaussian kernel suggested
    by [Haghverdi16]_ in the implementation of [Wolf18]_.
    The width ("sigma") of the connectivity kernel is implicitly determined by
    the number of neighbors used to compute the single-cell graph in
    :func:`~scanpy.pp.neighbors`. To reproduce the original implementation
    using a Gaussian kernel, use `method=='gauss'` in
    :func:`~scanpy.pp.neighbors`. To use an exponential kernel, use the default
    `method=='umap'`. Differences between these options shouldn't usually be
    dramatic.
    Parameters
    ----------
    adata
        Annotated data matrix.
    n_comps
        The number of dimensions of the representation.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    `X_diffmap` : :class:`numpy.ndarray` (`adata.obsm`)
        Diffusion map representation of data, which is the right eigen basis of
        the transition matrix with eigenvectors as columns.
    `diffmap_evals` : :class:`numpy.ndarray` (`adata.uns`)
        Array of size (number of eigen vectors).
        Eigenvalues of transition matrix.
    """

    scanpy.tl.diffmap(adata, n_comps=n_comps, copy=copy)

    print(
        "Diffusion Map is done! Generated in adata.obsm['X_diffmap'] nad adata.uns['diffmap_evals']")

    return adata if copy else None