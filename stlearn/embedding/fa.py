import numpy as np
import pandas as pd
from typing import Optional

from anndata import AnnData
from sklearn.decomposition import FactorAnalysis
from scipy.sparse import issparse


def run_fa(
    adata: AnnData,
    n_factors: int = 20,
    tol: float = 0.01,
    max_iter: int = 1000,
    svd_method: str = "randomized",
    iterated_power: int = 3,
    random_state: int = 2108,
    use_data: str = None,
    copy: bool = False,
) -> Optional[AnnData]:

    """\
    Factor Analysis (FA)
    A simple linear generative model with Gaussian latent variables.
    The observations are assumed to be caused by a linear transformation of
    lower dimensional latent factors and added Gaussian noise.
    Without loss of generality the factors are distributed according to a
    Gaussian with zero mean and unit covariance. The noise is also zero mean
    and has an arbitrary diagonal covariance matrix.
    If we would restrict the model further, by assuming that the Gaussian
    noise is even isotropic (all diagonal entries are the same) we would obtain
    :class:`PPCA`.
    FactorAnalysis performs a maximum likelihood estimate of the so-called
    `loading` matrix, the transformation of the latent variables to the
    observed ones, using SVD based approach.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_components
        Dimensionality of latent space, the number of components
        of ``X`` that are obtained after ``transform``.
        If None, n_components is set to the number of features.
    tol
        Stopping tolerance for log-likelihood increase.
    max_iter
        Maximum number of iterations.
    noise_variance_init
        The initial guess of the noise variance for each feature.
        If None, it defaults to np.ones(n_features)
    svd_method
        Which SVD method to use. If 'lapack' use standard SVD from
        scipy.linalg, if 'randomized' use fast ``randomized_svd`` function.
        Defaults to 'randomized'. For most applications 'randomized' will
        be sufficiently precise while providing significant speed gains.
        Accuracy can also be improved by setting higher values for
        `iterated_power`. If this is not sufficient, for maximum precision
        you should choose 'lapack'.
    iterated_power
        Number of iterations for the power method. 3 by default. Only used
        if ``svd_method`` equals 'randomized'
    random_state
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Only used when ``svd_method`` equals 'randomized'.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    `X_fa` : :class:`numpy.ndarray` (`adata.obsm`)
        Factor analysis representation of data.
    """

    if use_data is None:
        if issparse(adata.X):
            matrix = adata.X.toarray()
        else:
            matrix = adata.X

    else:
        matrix = adata.obsm[use_data].values

    fa = FactorAnalysis(n_components=n_factors, tol=tol, max_iter=max_iter,
                        svd_method=svd_method, iterated_power=iterated_power,
                        random_state=random_state)

    latent = fa.fit_transform(matrix)

    adata.obsm["X_fa"] = latent

    adata.uns['fa_params'] = {'params': {'n_factors': n_factors, 'tol': tol,
                                         'max_iter': max_iter, 'svd_method': svd_method,
                                         'iterated_power': iterated_power,
                                         'random_state': random_state}}

    print('FA is done! Generated in adata.obsm["X_fa"]')

    return adata if copy else None
