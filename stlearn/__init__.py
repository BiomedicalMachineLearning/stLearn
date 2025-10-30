"""Top-level package for stLearn."""

__author__ = """Genomics and Machine Learning Lab"""
__email__ = "andrew.newman@uq.edu.au"
__version__ = "1.2.2"

from . import add, datasets, em, pl, pp, spatial, tl, types
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
    "types",
    "convert_scanpy",
    "concatenate_spatial_adata",
]
