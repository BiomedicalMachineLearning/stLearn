"""Top-level package for stLearn."""

__author__ = """Genomics and Machine Learning Lab"""
__email__ = "andrew.newman@uq.edu.au"
__version__ = "1.5.0"

from . import add, datasets, em, io, pl, pp, spatial, tl, types, utils
from ._settings import settings
from .wrapper.concatenate_spatial_adata import concatenate_spatial_adata
from .wrapper.convert_scanpy import convert_scanpy

# from . import cli
__all__ = [
    "add",
    "concatenate_spatial_adata",
    "convert_scanpy",
    "datasets",
    "em",
    "io",
    "pl",
    "pp",
    "settings",
    "spatial",
    "tl",
    "types",
    "utils",
]
