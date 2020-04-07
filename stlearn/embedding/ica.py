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
