from anndata import AnnData

from .global_level import global_level
from .local_level import local_level
from .weight_optimization import weight_optimizing_global, weight_optimizing_local


def pseudotimespace_global(
    adata: AnnData,
    use_label: str = "louvain",
    use_rep: str = "X_pca",
    n_dims: int = 40,
    list_clusters=None,
    model: str = "spatial",
    step=0.01,
    k=10,
) -> AnnData | None:
    """\
    Perform pseudo-time-space analysis with global level.

    Parameters
    ----------
    adata:
        Annotated data matrix.
    use_label:
        Use label result of cluster method.
    use_rep:
        Which obsm location to use.
    n_dims:
        Number of dimensions to use in PCA
    list_clusters:
        List of cluster used to reconstruct spatial trajectory.
    model:
        Can be mixed, spatial or gene expression. spatial sets weight to 0,
        gene expression sets weight to 1 and mixed uses the list_clusters, step and k.
    step:
        Step for screening weighting factor
    k
        The number of eigenvalues to be compared
    Returns
    -------
    Anndata
    """

    if list_clusters is None:
        list_clusters = []

    if model == "mixed":
        w = weight_optimizing_global(
            adata, use_label=use_label, list_clusters=list_clusters, step=step, k=k
        )
    elif model == "spatial":
        w = 0
    elif model == "gene_expression":
        w = 1
    else:
        raise ValueError(
            "Please choose the right model! Available models: 'mixed', 'spatial' "
            + "and 'gene_expression' "
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
    cluster=None,
    w: float = None,
) -> AnnData | None:
    """\
    Perform pseudo-time-space analysis with local level.

    Parameters
    ----------
    adata:
        Annotated data matrix.
    use_label:
        Use label result of cluster method.
    cluster:
        Cluster used to reconstruct intra regional spatial trajectory.
    w:
        Weighting factor to balance between spatial data and gene expression
    Returns
    -------
    Anndata
    """

    if cluster is None:
        cluster = []
    if w is None:
        w = weight_optimizing_local(adata, use_label=use_label, cluster=cluster)

    local_level(adata, use_label=use_label, cluster=cluster, w=w)
