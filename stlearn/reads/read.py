"""Reading and Writing
"""
from pathlib import Path, PurePath
from typing import Union

from anndata import AnnData


def file_10x_h5(
    filename: Union[str, Path],
) -> AnnData:
    """
    Wrap function scanpy.read_10x_h5
    """

    from stlearn.external.scanpy.readwrite import read_10x_h5

    adata = read_10x_h5(filename)

    return adata


def file_10x_mtx(
    filename: Union[str, Path],
) -> AnnData:
    """
    Wrap function scanpy.read_10x_mtx
    """

    from stlearn.external.scanpy.readwrite import read_10x_mtx

    adata = read_10x_mtx(filename)

    return adata


def file_table(
    filename: Union[str, Path],
) -> AnnData:
    """
    Wrap function scanpy.read
    """

    from stlearn.external.scanpy.readwrite import read

    adata = read(filename)

    return adata
