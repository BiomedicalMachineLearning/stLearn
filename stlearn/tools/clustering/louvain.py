from types import MappingProxyType
from typing import Optional, Tuple, Sequence, Type, Mapping, Any, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from natsort import natsorted
from numpy.random.mtrand import RandomState
from scipy.sparse import spmatrix
from stlearn._compat import Literal

try:
    from louvain.VertexPartition import MutableVertexPartition
except ImportError:
    class MutableVertexPartition: pass
    MutableVertexPartition.__module__ = 'louvain.VertexPartition'
import scanpy

def louvain(
    adata: AnnData,
    resolution: Optional[float] = None,
    random_state: Optional[Union[int, RandomState]] = 0,
    restrict_to: Optional[Tuple[str, Sequence[str]]] = None,
    key_added: str = 'louvain',
    adjacency: Optional[spmatrix] = None,
    flavor: Literal['vtraag', 'igraph', 'rapids'] = 'vtraag',
    directed: bool = True,
    use_weights: bool = False,
    partition_type: Optional[Type[MutableVertexPartition]] = None,
    partition_kwargs: Mapping[str, Any] = MappingProxyType({}),
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Wrap function scanpy.tl.louvain
    Cluster cells into subgroups [Blondel08]_ [Levine15]_ [Traag17]_.
    Cluster cells using the Louvain algorithm [Blondel08]_ in the implementation
    of [Traag17]_. The Louvain algorithm has been proposed for single-cell
    analysis by [Levine15]_.
    This requires having ran :func:`~scanpy.pp.neighbors` or
    :func:`~scanpy.external.pp.bbknn` first,
    or explicitly passing a ``adjacency`` matrix.
    Parameters
    ----------
    adata
        The annotated data matrix.
    resolution
        For the default flavor (``'vtraag'``), you can provide a resolution
        (higher resolution means finding more and smaller clusters),
        which defaults to 1.0.
        See “Time as a resolution parameter” in [Lambiotte09]_.
    random_state
        Change the initialization of the optimization.
    restrict_to
        Restrict the clustering to the categories within the key for sample
        annotation, tuple needs to contain ``(obs_key, list_of_categories)``.
    key_added
        Key under which to add the cluster labels. (default: ``'louvain'``)
    adjacency
        Sparse adjacency matrix of the graph, defaults to
        ``adata.uns['neighbors']['connectivities']``.
    flavor
        Choose between to packages for computing the clustering.
        ``'vtraag'`` is much more powerful, and the default.
    directed
        Interpret the ``adjacency`` matrix as directed graph?
    use_weights
        Use weights from knn graph.
    partition_type
        Type of partition to use.
        Only a valid argument if ``flavor`` is ``'vtraag'``.
    partition_kwargs
        Key word arguments to pass to partitioning,
        if ``vtraag`` method is being used.
    copy
        Copy adata or modify it inplace.
    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obs['louvain']`` (:class:`pandas.Series`, dtype ``category``)
            Array of dim (number of samples) that stores the subgroup id
            (``'0'``, ``'1'``, ...) for each cell.
    :class:`~anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """

    scanpy.tl.louvain(adata, resolution=resolution, random_state=random_state, 
            restrict_to=restrict_to, key_added=key_added, adjacency=adjacency,
            flavor=flavor, directed=directed, use_weights=use_weights, partition_type=partition_type,
            partition_kwargs=partition_kwargs, copy=copy)

    print("Applying Louvain clustering ...")
    print("Louvain clustering is done! The labels are stored in adata.obs['louvain']")