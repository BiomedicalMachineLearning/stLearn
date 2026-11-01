"""Shared steps of SME normalisation and imputation."""

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse


def expression(adata: AnnData, use_data: str) -> sparse.csr_matrix | np.ndarray:
    """The data SME adjusts: `adata.X` for "raw", else `adata.obsm[use_data]`.

    Sparse data stays sparse (as CSR); nothing is densified.
    """
    data = adata.X if use_data == "raw" else adata.obsm[use_data]
    if sparse.issparse(data):
        return sparse.csr_matrix(data)
    if isinstance(data, pd.DataFrame):
        return data.to_numpy()
    if isinstance(data, np.ndarray):
        return data
    raise ValueError(f"{type(data)} is not a valid type.")


def average_with_imputed(
    data: sparse.csr_matrix | np.ndarray, imputed: sparse.csr_matrix | np.ndarray
) -> sparse.csr_matrix | np.ndarray:
    """The mean of `data` and `imputed` where `imputed` is not 0, else `data`."""
    if sparse.issparse(data):
        imputed = sparse.csr_matrix(imputed)
        imputed.eliminate_zeros()
        on_imputed = data.multiply(imputed != 0)
        result = data - on_imputed + (on_imputed + imputed) / 2
        result = sparse.csr_matrix(result, dtype=np.float64)
        result.eliminate_zeros()
        return result
    # Written into `imputed`, a new array, so only a mask is added.
    result = np.asarray(imputed, dtype=np.float64)
    nonzero = result != 0
    result[nonzero] += data[nonzero]
    result[nonzero] /= 2
    result[~nonzero] = data[~nonzero]
    return result


def fill_zeros(
    data: sparse.csr_matrix | np.ndarray, imputed: sparse.csr_matrix | np.ndarray
) -> sparse.csr_matrix | np.ndarray:
    """`data`, with its zeros replaced by `imputed`."""
    if sparse.issparse(data):
        imputed = sparse.csr_matrix(imputed)
        result = data + (imputed - imputed.multiply(data != 0))
        result = sparse.csr_matrix(result, dtype=np.float64)
        result.eliminate_zeros()
        return result
    result = np.asarray(imputed, dtype=np.float64)
    nonzero = data != 0
    result[nonzero] = data[nonzero]
    return result


def adjust_by_blocks(
    combine,
    data: sparse.csr_matrix | np.ndarray,
    neighbours: sparse.csr_matrix,
    spots_per_block: int = 10000,
) -> sparse.csr_matrix | np.ndarray:
    """`combine(data, neighbours @ data)`, a block of spots at a time, so the
    imputed data and the temporaries for combining it are only ever made for
    one block."""
    n_spots = data.shape[0]
    if n_spots <= spots_per_block:
        return combine(data, neighbours @ data)
    blocks = range(0, n_spots, spots_per_block)
    if sparse.issparse(data):
        return sparse.vstack(
            [
                combine(
                    data[start : start + spots_per_block],
                    neighbours[start : start + spots_per_block] @ data,
                )
                for start in blocks
            ],
            format="csr",
        )
    result = np.empty(data.shape, dtype=np.float64)
    for start in blocks:
        stop = start + spots_per_block
        result[start:stop] = combine(data[start:stop], neighbours[start:stop] @ data)
    return result
