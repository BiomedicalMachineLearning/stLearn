from typing import Literal

from anndata import AnnData

from stlearn.utils import _check_cluster, _cluster_labels

from .global_level import global_level
from .local_level import local_level
from .weight_optimization import weight_optimizing_global, weight_optimizing_local


def pseudotimespace_global(
    adata: AnnData,
    use_label: str = "leiden",
    use_rep: str = "X_pca",
    n_dims: int = 40,
    list_clusters: str | list[str] | None = None,
    model: Literal["spatial", "gene_expression", "mixed"] = "spatial",
    step=0.01,
    k=10,
) -> AnnData | None:
    """\
    Perform pseudo-time-space analysis with global level.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    use_label: str, default = "leiden"
        Use label result of cluster method.
    use_rep: str, default = "X_pca"
        Which obsm location to use.
    n_dims: int, default = 40
        Number of dimensions to use in PCA
    list_clusters: list of str, optional
         Labels of the clusters in ``adata.obs[use_label]`` to reconstruct the spatial
         trajectory between, e.g. ``["6", "9"]``. If None, uses all clusters.
    model: Literal["spatial", "gene_expression", "mixed"] = "mixed",
        Can be mixed, spatial or gene expression. spatial sets weight to 0,
        gene expression sets weight to 1 and mixed uses the list_clusters, step and k.
    step: float, default = 0.01
        Step for screening weighting factor.
    k: int, default = 10
        The number of eigenvalues to be compared.
    Returns
    -------
    Anndata
    """

    labels = _cluster_labels(
        adata, use_label, [] if list_clusters is None else list_clusters
    )

    if model == "mixed":
        w = weight_optimizing_global(
            adata, use_label=use_label, list_clusters=labels, step=step, k=k
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
        list_clusters=labels,
        w=w,
        use_rep=use_rep,
        n_dims=n_dims,
    )

    return adata


def pseudotimespace_local(
    adata: AnnData,
    use_label: str = "leiden",
    *,
    cluster: str,
    w: float | None = None,
) -> AnnData | None:
    """\
    Perform pseudo-time-space analysis with local level.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    use_label: str, default = "leiden"
        Use label result of cluster method.
    cluster:
        Label of the cluster in ``adata.obs[use_label]`` used to reconstruct the
        intra-regional spatial trajectory, e.g. ``"6"``.
    w:
        Weighting factor to balance between spatial data and gene expression
    Returns
    -------
    Anndata
    """

    cluster = _check_cluster(adata, use_label, cluster)

    if w is None:
        w = weight_optimizing_local(adata, use_label=use_label, cluster=cluster)

    local_level(adata, use_label=use_label, cluster=cluster, w=w)

    return adata
