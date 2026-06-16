"""Top-level package for stLearn."""

__author__ = """Genomics and Machine Learning Lab"""
__email__ = "andrew.newman@uq.edu.au"
__version__ = "1.4.1"

from . import add, datasets, em, pl, pp, spatial, tl, types
from ._settings import settings
from .wrapper.concatenate_spatial_adata import concatenate_spatial_adata
from .wrapper.convert_scanpy import convert_scanpy

# Wrapper
from .wrapper.read import (
    create_stlearn,
    read_10x,
    read_merfish,
    read_old_st,
    read_seq_fish,
    read_slide_seq,
    read_xenium,
)

# from . import cli
__all__ = [
    "add",
    "concatenate_spatial_adata",
    "convert_scanpy",
    "create_stlearn",
    "datasets",
    "em",
    "pl",
    "pp",
    "read_10x",
    "read_merfish",
    "read_old_st",
    "read_seq_fish",
    "read_slide_seq",
    "read_xenium",
    "settings",
    "spatial",
    "tl",
    "types",
]
