from typing import Optional, Union, Iterable, Dict

import numpy as np
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs
from stlearn._compat import Literal
import scanpy

def normalize_total(
    adata: AnnData,
    target_sum: Optional[float] = None,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    key_added: Optional[str] = None,
    layers: Union[Literal['all'], Iterable[str]] = None,
    layer_norm: Optional[str] = None,
    inplace: bool = True,
) -> Optional[Dict[str, np.ndarray]]:
    """\
    Wrap function from scanpy.pp.log1p
    Normalize counts per cell.
    If choosing `target_sum=1e6`, this is CPM normalization.
    If `exclude_highly_expressed=True`, very highly expressed genes are excluded
    from the computation of the normalization factor (size factor) for each
    cell. This is meaningful as these can strongly influence the resulting
    normalized values for all other genes [Weinreb17]_.
    Similar functions are used, for example, by Seurat [Satija15]_, Cell Ranger
    [Zheng17]_ or SPRING [Weinreb17]_.
    Params
    ------
    adata
        The annotated data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    target_sum
        If `None`, after normalization, each observation (cell) has a total
        count equal to the median of total counts for observations (cells)
        before normalization.
    exclude_highly_expressed
        Exclude (very) highly expressed genes for the computation of the
        normalization factor (size factor) for each cell. A gene is considered
        highly expressed, if it has more than `max_fraction` of the total counts
        in at least one cell. The not-excluded genes will sum up to
        `target_sum`.
    max_fraction
        If `exclude_highly_expressed=True`, consider cells as highly expressed
        that have more counts than `max_fraction` of the original total counts
        in at least one cell.
    key_added
        Name of the field in `adata.obs` where the normalization factor is
        stored.
    layers
        List of layers to normalize. Set to `'all'` to normalize all layers.
    layer_norm
        Specifies how to normalize layers:
        * If `None`, after normalization, for each layer in *layers* each cell
          has a total count equal to the median of the *counts_per_cell* before
          normalization of the layer.
        * If `'after'`, for each layer in *layers* each cell has
          a total count equal to `target_sum`.
        * If `'X'`, for each layer in *layers* each cell has a total count
          equal to the median of total counts for observations (cells) of
          `adata.X` before normalization.
    inplace
        Whether to update `adata` or return dictionary with normalized copies of
        `adata.X` and `adata.layers`.
    Returns
    -------
    Returns dictionary with normalized copies of `adata.X` and `adata.layers`
    or updates `adata` with normalized version of the original
    `adata.X` and `adata.layers`, depending on `inplace`.
    """


    scanpy.pp.normalize_total(adata, target_sum=target_sum,
                    exclude_highly_expressed=exclude_highly_expressed,
                    max_fraction=max_fraction, key_added=key_added,
                    layers=layers, layer_norm=layer_norm, inplace=inplace)

    adata.obsm["normalized_total"] = adata.X

    print("Normalization step is finished in adata.X")
