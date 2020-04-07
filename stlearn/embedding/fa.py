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
