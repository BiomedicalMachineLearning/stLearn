"""Top-level package for stLearn."""

__author__ = """Genomics and Machine Learning lab"""
__email__ = "andrew.newman@uq.edu.au"
__version__ = "0.5.0"

from . import add, datasets, em, pl, pp, spatial, tl
from ._settings import settings
from .wrapper.concatenate_spatial_adata import concatenate_spatial_adata
from .wrapper.convert_scanpy import convert_scanpy

# Wrapper
from .wrapper.read import (
    Read10X,
    ReadMERFISH,
    ReadOldST,
    ReadSeqFish,
    ReadSlideSeq,
    ReadXenium,
    create_stlearn,
)

# from . import cli
__all__ = [
    "add",
    "pp",
    "em",
    "tl",
    "pl",
    "spatial",
    "datasets",
    "ReadSlideSeq",
    "Read10X",
    "ReadOldST",
    "ReadMERFISH",
    "ReadSeqFish",
    "ReadXenium",
    "create_stlearn",
    "settings",
    "convert_scanpy",
    "concatenate_spatial_adata",
]
