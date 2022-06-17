from anndata import AnnData
from typing import Optional, Union
import numpy as np
from stlearn.spatials.trajectory.utils import _correlation_test_helper


def set_root(adata: AnnData, use_label: str, cluster: str, use_raw: bool = False):

    """\
    Automatically set the root index.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_label
        Use label result of cluster method.
    cluster
        Choose cluster to use as root
    use_raw
        Use the raw layer
    Returns
    -------
    Root index
    """

    tmp_adata = adata.copy()

    # Subset the data based on the chosen cluster

    tmp_adata = tmp_adata[
        tmp_adata.obs[tmp_adata.obs[use_label] == str(cluster)].index, :
    ]
    if use_raw == True:
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
