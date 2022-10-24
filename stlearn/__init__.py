"""Top-level package for stLearn."""

__author__ = """Genomics and Machine Learning lab"""
__email__ = "duy.pham@uq.edu.au"
__version__ = "0.4.8"


from . import add
from . import pp
from . import em
from . import tl
from . import pl
from . import spatial
from . import datasets

# Wrapper

from .wrapper.read import ReadSlideSeq
from .wrapper.read import Read10X
from .wrapper.read import ReadOldST
from .wrapper.read import ReadMERFISH
from .wrapper.read import ReadSeqFish
from .wrapper.read import ReadXenium
from .wrapper.read import create_stlearn

from ._settings import settings
from .wrapper.convert_scanpy import convert_scanpy
from .wrapper.concatenate_spatial_adata import concatenate_spatial_adata

# from . import cli
