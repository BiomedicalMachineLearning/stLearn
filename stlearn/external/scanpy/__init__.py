
# the actual API
from ._settings import settings, Verbosity  # start with settings as several tools are using it
from . import tools as tl
from . import preprocessing as pp
from . import plotting as pl
from . import logging, queries, external, get

from anndata import AnnData
from anndata import read_h5ad, read_csv, read_excel, read_hdf, read_loom, read_mtx, read_text, read_umi_tools
from .readwrite import read, read_10x_h5, read_10x_mtx, write
from .neighbors import Neighbors

set_figure_params = settings.set_figure_params