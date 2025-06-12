import numpy as np
import scanpy
from anndata import AnnData
from scipy.sparse import spmatrix


def log1p(
    adata: AnnData | np.ndarray | spmatrix,
    copy: bool = False,
    chunked: bool = False,
    chunk_size: int | None = None,
    base: float | None = None,
) -> AnnData | None:
    """\
    Wrap function of scanpy.pp.log1p
    Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab

    Logarithmize the data matrix.
    Computes :math:`X = \\log(X + 1)`,
    where :math:`log` denotes the natural logarithm unless a different base is given.
    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    copy
        If an :class:`~anndata.AnnData` is passed, determines whether a copy
        is returned.
    chunked
        Process the data matrix in chunks, which will save memory.
        Applies only to :class:`~anndata.AnnData`.
    chunk_size
        `n_obs` of the chunks to process the data in.
    base
        Base of the logarithm. Natural logarithm is used by default.
    Returns
    -------
    Returns or updates `data`, depending on `copy`.
    """

    result = scanpy.pp.log1p(
        adata, copy=copy, chunked=chunked, chunk_size=chunk_size, base=base
    )
    print("Log transformation step is finished in adata.X")
    return result


def scale(
    data: AnnData | spmatrix | np.ndarray,
    zero_center: bool = True,
    max_value: float | None = None,
    copy: bool = False,
) -> AnnData | spmatrix | np.ndarray | None:
    """\
    Wrap function of scanpy.pp.scale

    Scale data to unit variance and zero mean.
    .. note::
        Variables (genes) that do not display any variation (are constant across
        all observations) are retained and set to 0 during this operation. In
        the future, they might be set to NaNs.
    Parameters
    ----------
    data:
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    zero_center
        If `False`, omit zero-centering variables, which allows to handle sparse
        input efficiently.
    max_value
        Clip (truncate) to this value after scaling. If `None`, do not clip.
    copy
        If an :class:`~anndata.AnnData` is passed,
        determines whether a copy is returned.
    Returns
    -------
    Depending on `copy` returns or updates `data` with a scaled `data.X`.
    """

    result = scanpy.pp.scale(
        data, zero_center=zero_center, max_value=max_value, copy=copy
    )
    print("Scale step is finished in adata.X")
    return result
