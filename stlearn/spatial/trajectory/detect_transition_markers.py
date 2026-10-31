import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import spearmanr

from ...utils import _check_cluster, _cluster_labels, _read_trajectory


def detect_transition_markers_clades(
    adata: AnnData,
    clade: str,
    cutoff_spearman: float = 0.4,
    cutoff_pvalue: float = 0.05,
    screening_genes: list[str] | None = None,
    use_raw_count: bool = False,
) -> None:
    """\
    Transition markers detection of a clade.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing spatial transcriptomics data with
        computed pseudotime and clade information.
    clade : str
        The subcluster the clade starts from, as a label in
        ``adata.obs["sub_cluster_labels"]``, e.g. ``"17"``. The clade is that
        subcluster and the subclusters it leads to in the spatial trajectory.
    cutoff_spearman : float, default 0.4
        The minimum Spearman correlation coefficient threshold for identifying
        significant gene-pseudotime correlations. Must be between 0 and 1.
    cutoff_pvalue : float, default 0.05
        The maximum p-value threshold for statistical significance testing.
        Must be between 0 and 1. Lower values result in more stringent
        statistical filtering.
    screening_genes : list of str, optional
        Custom list of gene names to restrict the analysis to. If None,
        all genes in the dataset will be considered. Useful for focusing
        on specific gene sets or reducing computational time.
    use_raw_count : bool, default False
        True if user wants to use raw layer data.
    Returns
    -------
    AnnData
        The input AnnData object with additional information stored in
        adata.uns about the detected transition markers, including:
        - Correlation coefficients
        - P-values
        - Gene rankings
        - Clade-specific marker information
    """

    clade = _check_cluster(adata, "sub_cluster_labels", clade)
    clade_key = f"clade_{clade}"
    print(f"Detecting the transition markers of {clade_key}...")

    G = _read_trajectory(adata)
    clades = [node for node in G if G.out_degree(node) > 0]
    if clade not in clades:
        raise ValueError(
            f"Subcluster {clade!r} does not start a clade in the spatial "
            f"trajectory. Choose from: {clades}"
        )
    nodes = [clade, *G.successors(clade)]

    query_adata = adata[adata.obs["sub_cluster_labels"].isin(nodes)]

    spearman_result = get_rank_cor(
        query_adata, screening_genes=screening_genes, use_raw_count=use_raw_count
    )

    spearman_result = spearman_result[spearman_result["p-value"] < cutoff_pvalue]
    positive = spearman_result[spearman_result["score"] >= cutoff_spearman].sort_values(
        "score", ascending=False
    )
    negative = spearman_result[
        spearman_result["score"] <= cutoff_spearman * (-1)
    ].sort_values("score")

    result = pd.concat([positive, negative])
    adata.uns[clade_key] = result
    print(f"Transition markers result is stored in adata.uns['{clade_key}']")


def detect_transition_markers_branches(
    adata: AnnData,
    branch: list[str],
    cutoff_spearman: float = 0.4,
    cutoff_pvalue: float = 0.05,
    screening_genes: list[str] | None = None,
    use_raw_count: bool = False,
) -> None:
    """\
    Transition markers detection of a branch.

    Parameters
    ----------
    adata:
        Annotated data matrix.
    branch:
        The subclusters of the branch, as labels in
        ``adata.obs["sub_cluster_labels"]``, e.g. ``["7", "8"]``.
    cutoff_spearman:
        The threshold of correlation coefficient.
    cutoff_pvalue:
        The threshold of p-value.
    screening_genes:
        List of customised genes.
    use_raw_count:
        True if user wants to use raw layer data.
    Returns
    -------
    Anndata
    """
    branch = _cluster_labels(adata, "sub_cluster_labels", branch)
    branch_key = f"branch_{'_'.join(branch)}"

    print(f"Detecting the transition markers of {branch_key}")

    query_adata = adata[adata.obs["sub_cluster_labels"].isin(branch)]

    spearman_result = get_rank_cor(
        query_adata, screening_genes=screening_genes, use_raw_count=use_raw_count
    )

    spearman_result = spearman_result[spearman_result["p-value"] < cutoff_pvalue]
    positive = spearman_result[spearman_result["score"] >= cutoff_spearman].sort_values(
        "score", ascending=False
    )
    negative = spearman_result[
        spearman_result["score"] <= cutoff_spearman * (-1)
    ].sort_values("score")

    result = pd.concat([positive, negative])
    adata.uns[branch_key] = result
    print(f"Transition markers result is stored in adata.uns['{branch_key}']")


def get_rank_cor(
    adata: AnnData,
    screening_genes: list[str] | None = None,
    use_raw_count: bool = True,
) -> pd.DataFrame:
    if use_raw_count:
        tmp = adata.copy()
        tmp.X = adata.layers["raw_count"]
        tmp = tmp.to_df()
    else:
        tmp = adata.to_df()

    if screening_genes is not None:
        tmp = tmp[screening_genes]

    dpt = adata.obs["dpt_pseudotime"].values
    genes = []
    score = []
    pvalue = []

    for gene in tmp.columns:
        gene_values = tmp[gene].values
        if np.std(gene_values) == 0 or np.std(dpt) == 0:
            genes.append(gene)
            score.append(np.nan)
            pvalue.append(np.nan)
        else:
            result = spearmanr(gene_values, dpt)
            genes.append(gene)
            score.append(result[0])
            pvalue.append(result[1])
    return pd.DataFrame({"gene": genes, "score": score, "p-value": pvalue})
