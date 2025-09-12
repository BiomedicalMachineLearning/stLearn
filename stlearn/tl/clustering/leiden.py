from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

import scanpy
from anndata import AnnData
from louvain.VertexPartition import MutableVertexPartition
from numpy.random.mtrand import RandomState
from scipy.sparse import spmatrix


def louvain(
    adata: AnnData,
    resolution: float | None = None,
    random_state: int | RandomState | None = 0,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    key_added: str = "leiden",
    adjacency: spmatrix | None = None,
    directed: bool = True,
    use_weights: bool = False,
    partition_type: type[MutableVertexPartition] | None = None,
    obsp: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """\
    Wrap function scanpy.tl.leiden

    This requires having ran :func:`~scanpy.pp.neighbors` or
    :func:`~scanpy.external.pp.bbknn` first,
    or explicitly passing a ``adjacency`` matrix.
    Parameters
    ----------
    adata:
        The annotated data matrix.
    resolution:
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
        Set to `None` if overriding `partition_type`
        to one that doesn’t accept a `resolution_parameter`.
    random_state:
        Change the initialization of the optimization.
    restrict_to:
        Restrict the cluster to the categories within the key for sample
        annotation, tuple needs to contain ``(obs_key, list_of_categories)``.
    key_added:
        Key under which to add the cluster labels. (default: ``'leiden'``)
    adjacency:
        Sparse adjacency matrix of the graph, defaults to
        ``adata.uns['neighbors']['connectivities']``.
    directed:
        Interpret the ``adjacency`` matrix as directed graph?
    use_weights:
        Use weights from knn graph.
    partition_type:
        Type of partition to use.
        Defaults to :class:`~leidenalg.RBConfigurationVertexPartition`.
        For the available options, consult the documentation for
        :func:`~leidenalg.find_partition`.
    obsp:
        Use .obsp[obsp] as adjacency. You can't specify both
        `obsp` and `neighbors_key` at the same time.
    copy:
        Copy adata or modify it inplace.
    Returns
    -------
    :obj:`None`
        By default (``copy=False``), updates ``adata`` with the following fields:
        ``adata.obs['leiden' | key_added]`` (:class:`pandas.Series`, dtype ``category``)
            Array of dim (number of samples) that stores the subgroup id
            (``'0'``, ``'1'``, ...) for each cell.
    :class:`~anndata.AnnData`
        When ``copy=True`` is set, a copy of ``adata`` with those fields is returned.
    """

    print("Applying Leiden cluster ...")
    adata = scanpy.tl.leiden(
        adata,
        resolution=resolution,
        restrict_to=restrict_to,
        random_state=random_state,
        key_added=key_added,
        adjacency=adjacency,
        directed=directed,
        use_weights=use_weights,
        partition_type=partition_type,
        obsp=obsp,
        copy=copy,
    )

    print(
        "Leiden cluster is done! The labels are stored in adata.obs['%s']" % key_added
    )

    return adata
