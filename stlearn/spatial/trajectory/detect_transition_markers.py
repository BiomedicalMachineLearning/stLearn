import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import spearmanr

from ...utils import _read_graph


def detect_transition_markers_clades(
    adata: AnnData,
    clade: int,
    cutoff_spearman: float = 0.4,
    cutoff_pvalue: float = 0.05,
    screening_genes: None | list[str] = None,
    use_raw_count: bool = False,
) -> None:
    """\
    Transition markers detection of a clade.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing spatial transcriptomics data with
        computed pseudotime and clade information.
    clade : int
        Numeric identifier of the clade for which to detect transition markers.
        Should correspond to a clade ID present in the trajectory analysis.
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

    clade_key = f"clade_{clade}"
    print(f"Detecting the transition markers of {clade_key}...")

    tmp = _read_graph(adata, "PTS_graph")
    G = tmp.copy()
    remove = [edge for edge in G.edges if 9999 in edge]
    G.remove_edges_from(remove)
    G.remove_node(9999)

    nodes = []
    for edge in G.edges([clade]):
        nodes.append(edge[0])
        nodes.append(edge[1])
    nodes = list(set(nodes))

    assert clade in nodes, "Please choose the right clade!"

    query_adata = adata[adata.obs.query(create_query(nodes)).index]

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
    branch: list[int],
    cutoff_spearman: float = 0.4,
    cutoff_pvalue: float = 0.05,
    screening_genes: None | list[str] = None,
    use_raw_count: bool = False,
) -> None:
    """\
    Transition markers detection of a branch.

    Parameters
    ----------
    adata
        Annotated data matrix.
    branch
        Name of a branch user wants to detect transition markers.
    cutoff_spearman
        The threshold of correlation coefficient.
    cutoff_pvalue
        The threshold of p-value.
    screening_genes
        List of customised genes.
    use_raw_count
        True if user wants to use raw layer data.
    Returns
    -------
    Anndata
    """
    branch_key = f"branch_{'_'.join(np.array(branch).astype(str))}"
    print(f"Detecting the transition markers of {branch_key}")

    query_adata = adata[adata.obs.query(create_query(branch)).index]

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


def create_query(list_sub_clusters: list) -> str:
    return " | ".join(f'sub_cluster_labels == "{sub}"' for sub in list_sub_clusters)


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

    for gene in list(adata.var.index):
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
