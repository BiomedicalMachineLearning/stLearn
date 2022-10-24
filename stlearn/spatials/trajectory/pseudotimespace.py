from anndata import AnnData
from typing import Optional, Union
from .weight_optimization import weight_optimizing_global, weight_optimizing_local
from .global_level import global_level
from .local_level import local_level


def pseudotimespace_global(
    adata: AnnData,
    use_label: str = "louvain",
    use_rep: str = "X_pca",
    n_dims: int = 40,
    list_clusters: list = [],
    model: str = "spatial",
    step=0.01,
    k=10,
) -> Optional[AnnData]:

    """\
    Perform pseudo-time-space analysis with global level.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_label
        Use label result of cluster method.
    list_clusters
        List of cluster used to reconstruct spatial trajectory.
    w
        Weighting factor to balance between spatial data and gene expression
    step
        Step for screeing weighting factor
    k
        The number of eigenvalues to be compared
    Returns
    -------
    Anndata
    """

    if model == "mixed":

        w = weight_optimizing_global(
            adata, use_label=use_label, list_clusters=list_clusters, step=step, k=k
        )
    elif model == "spatial":
        w = 0
    elif model == "gene_expression":
        w = 1
    else:
        raise ValidationError(
            "Please choose the right model! Available models: 'mixed', 'spatial' and 'gene_expression' "
        )

    global_level(
        adata,
        use_label=use_label,
        list_clusters=list_clusters,
        w=w,
        use_rep=use_rep,
        n_dims=n_dims,
    )


def pseudotimespace_local(
    adata: AnnData,
    use_label: str = "louvain",
    cluster: list = [],
    w: float = None,
) -> Optional[AnnData]:

    """\
    Perform pseudo-time-space analysis with local level.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_label
        Use label result of cluster method.
    cluster
        Cluster used to reconstruct intraregional spatial trajectory.
    w
        Weighting factor to balance between spatial data and gene expression
    Returns
    -------
    Anndata
    """

    if w is None:
        w = weight_optimizing_local(adata, use_label=use_label, cluster=cluster)

    local_level(adata, use_label=use_label, cluster=cluster, w=w)
