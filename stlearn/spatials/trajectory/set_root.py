import numpy as np
from anndata import AnnData

from stlearn.spatials.trajectory.utils import _correlation_test_helper


def set_root(adata: AnnData, use_label: str, cluster: str, use_raw: bool = False):
    """\
    Automatically set the root index for trajectory analysis.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    use_label: str
        Use label result of cluster method.
    cluster: str
        Cluster identifier to use as the root cluster. Must exist in
        `adata.obs[use_label]`. Will be converted to string for comparison.
    use_raw: bool, default False
        If True, use `adata.raw.X` for calculations; otherwise use `adata.X`.
    Returns
    -------
    int
        Index of the selected root cell in the AnnData object
    Raises
    ------
    ValueError
        If the specified cluster is not found in the clustering results.
    ZeroDivisionError
        If the specified cluster contains no cells.
    """

    tmp_adata = adata.copy()

    # Subset the data based on the chosen cluster
    available_clusters = tmp_adata.obs[use_label].unique()
    if str(cluster) not in available_clusters.astype(str):
        raise ValueError(
            f"Cluster '{cluster}' not found in available clusters: "
            + "{sorted(available_clusters)}"
        )

    tmp_adata = tmp_adata[
        tmp_adata.obs[tmp_adata.obs[use_label] == str(cluster)].index, :
    ]
    if use_raw:
        tmp_adata = tmp_adata.raw.to_adata()

    # Borrow from Cellrank to calculate CytoTrace score
    num_exp_genes = np.array((tmp_adata.X > 0).sum(axis=1)).reshape(-1)
    gene_corr, _, _, _ = _correlation_test_helper(tmp_adata.X.T, num_exp_genes[:, None])
    tmp_adata.var["gene_corr"] = gene_corr

    # Use top 1000 genes rather than top 200 genes
    top_1000 = tmp_adata.var.sort_values(by="gene_corr", ascending=False).index[:1000]
    tmp_adata.var["correlates"] = False
    tmp_adata.var.loc[top_1000, "correlates"] = True
    corr_mask = tmp_adata.var["correlates"]
    imputed_exp = tmp_adata[:, corr_mask].X

    # Scale ct score
    cytotrace_score = np.mean(imputed_exp, axis=1)
    cytotrace_score -= np.min(cytotrace_score)
    cytotrace_score /= np.max(cytotrace_score)

    # Get the root index
    local_index = np.argmax(cytotrace_score)
    obs_name = tmp_adata.obs.iloc[local_index].name

    return np.where(adata.obs_names == obs_name)[0][0]
