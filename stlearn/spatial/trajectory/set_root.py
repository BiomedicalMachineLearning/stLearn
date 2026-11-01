import numpy as np
import pandas as pd
from anndata import AnnData

from stlearn.spatial.trajectory.utils import _correlation_test_helper
from stlearn.utils import _check_cluster


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
        Label of the root cluster in ``adata.obs[use_label]``, e.g. ``"6"``.
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

    cluster = _check_cluster(adata, use_label, cluster)
    # Only the root cluster's rows are read; nothing else is copied.
    rows = np.flatnonzero((adata.obs[use_label] == cluster).to_numpy())
    if use_raw:
        expression, genes = adata.raw.X[rows], adata.raw.var_names
    else:
        expression, genes = adata.X[rows], adata.var_names

    # Borrow from Cellrank to calculate CytoTrace score
    num_exp_genes = np.array((expression > 0).sum(axis=1)).reshape(-1)
    gene_corr, _, _, _ = _correlation_test_helper(expression.T, num_exp_genes[:, None])
    gene_corr = pd.Series(np.asarray(gene_corr).reshape(-1), index=genes)

    # Use top 1000 genes rather than top 200 genes
    top_1000 = gene_corr.sort_values(ascending=False).index[:1000]
    imputed_exp = expression[:, genes.isin(top_1000)]

    # Scale ct score
    cytotrace_score = np.mean(imputed_exp, axis=1)
    cytotrace_score -= np.min(cytotrace_score)
    cytotrace_score /= np.max(cytotrace_score)

    # Get the root index
    return int(rows[np.argmax(cytotrace_score)])
