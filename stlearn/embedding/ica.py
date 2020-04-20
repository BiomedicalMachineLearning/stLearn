import numpy as np
import pandas as pd
from typing import Optional
from anndata import AnnData
from sklearn.decomposition import FastICA
from scipy.sparse import issparse


def run_ica(
    adata: AnnData,
    n_factors: int = 20,
    fun: str = "logcosh",
    tol: float = 0.0001,
    use_data: str = None,
    copy: bool = False,
) -> Optional[AnnData]:

    """\
    FastICA: a fast algorithm for Independent Component Analysis.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_factors
        Number of components to use. If none is passed, all are used.
    fun
        The functional form of the G function used in the
        approximation to neg-entropy. Could be either 'logcosh', 'exp',
        or 'cube'.
        You can also provide your own function. It should return a tuple
        containing the value of the function, and of its derivative, in the
        point. Example:
        def my_g(x):
            return x ** 3, (3 * x ** 2).mean(axis=-1)
    tol
        Tolerance on update at each iteration.
    use_data
        if None, then using all the gene expression profile. Else, use
        the chosen data from adata.obsm.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    `X_ica` : :class:`numpy.ndarray` (`adata.obsm`)
        Independent Component Analysis representation of data.
    """

    if use_data is None:
        if issparse(adata.X):
            matrix = adata.X.toarray()
        else:
            matrix = adata.X

    else:
        matrix = adata.obsm[use_data].values

    ica = FastICA(n_components=n_factors, fun=fun, tol=tol)

    latent = ica.fit_transform(matrix)

    adata.obsm["X_ica"] = latent

    adata.uns['ica'] = {'params': {'n_factors': n_factors, 'fun': fun,
                                   'tol': tol}}

    print(
        "ICA is done! Generated in adata.obsm['X_ica'] and parameters in adata.uns['ica']")

    return adata if copy else None
