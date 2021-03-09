"""Reading and Writing
"""
from pathlib import Path, PurePath
from typing import Union

from anndata import AnnData


def file_10x_h5(
    filename: Union[str, Path],
) -> AnnData:

    """\
    Wrap function scanpy.read_10x_h5
    Read 10x-Genomics-formatted hdf5 file.

    Parameters
    ----------
    filename
        Path to the h5 file.
    Returns
    -------
    AnnData
    """

    from scanpy.readwrite import read_10x_h5

    adata = read_10x_h5(filename)

    return adata


def file_10x_mtx(
    filename: Union[str, Path],
) -> AnnData:
    """\
    Wrap function scanpy.read_10x_mtx
    Read .mtx file.

    Parameters
    ----------
    filename
        Path to the h5 file.
    Returns
    -------
    AnnData
    """

    from scanpy.readwrite import read_10x_mtx

    adata = read_10x_mtx(filename)

    return adata


def file_table(
    filename: Union[str, Path],
) -> AnnData:
    """
    Wrap function scanpy.read
    Read file and return AnnData object.

    Parameters
    ----------
    filename
        Path to the h5 file.
    Returns
    -------
    AnnData
    """

    from scanpy.readwrite import read

    adata = read(filename)

    return adata
