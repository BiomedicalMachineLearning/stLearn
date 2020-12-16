from anndata import AnnData
from typing import Optional, Union
from .weight_optimization import weight_optimizing_global, weight_optimizing_local
from .global_level import global_level
from .local_level import local_level


def pseudotimespace_global(
    adata: AnnData,
    use_label: str = "louvain",
    list_cluster: list = [],
    w: float = None,
    step=0.01,
    k=10,
) -> Optional[AnnData]:

    if w is None:

        w = weight_optimizing_global(
            adata, use_label=use_label, list_cluster=list_cluster, step=step, k=k
        )

    global_level(adata, use_label=use_label, list_cluster=list_cluster, w=w)


def pseudotimespace_local(
    adata: AnnData,
    use_label: str = "louvain",
    cluster: list = [],
    w: float = None,
) -> Optional[AnnData]:

    if w is None:
        w = weight_optimizing_local(adata, use_label=use_label, cluster=cluster)

    local_level(adata, use_label=use_label, cluster=cluster, w=w)
