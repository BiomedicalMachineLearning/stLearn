from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def detect_transition_markers_clades(
    adata,
    clade,
    cutoff_spearman=0.4,
    cutoff_pvalue=0.05,
    screening_genes=None,
    use_raw_count=False,
):

    print("Detecting the transition markers of clade_" + str(clade) + "...")

    tmp = adata.uns["PTS_graph"]

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

    adata.uns["clade_" + str(clade)] = result

    print(
        "Transition markers result is stored in adata.uns['clade_" + str(clade) + "']"
    )


def detect_transition_markers_branches(
    adata,
    branch,
    cutoff_spearman=0.4,
    cutoff_pvalue=0.05,
    screening_genes=None,
    use_raw_count=False,
):

    print(
        "Detecting the transition markers of branch_"
        + "_".join(np.array(branch).astype(str))
    )

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

    adata.uns["branch_" + "_".join(np.array(branch).astype(str))] = result

    print(
        "Transition markers result is stored in adata.uns['branch_"
        + "_".join(np.array(branch).astype(str))
        + "']"
    )


def create_query(list_sub_clusters):
    ini = ""
    for sub in list_sub_clusters:
        ini = ini + 'sub_cluster_labels == "' + str(sub) + '" | '
    return ini[:-2]


def get_rank_cor(adata, screening_genes=None, use_raw_count=True):
    if use_raw_count:
        tmp = adata.copy()
        tmp.X = adata.layers["raw_count"]
        tmp = tmp.to_df()
    else:
        tmp = adata.to_df()
    if screening_genes != None:
        tmp = tmp[screening_genes]
    dpt = adata.obs["dpt_pseudotime"].values
    genes = []
    score = []
    pvalue = []
    for gene in list(adata.var.index):
        genes.append(gene)
        score.append(spearmanr(tmp[gene].values, dpt)[0])
        pvalue.append(spearmanr(tmp[gene].values, dpt)[1])
    import pandas as pd

    final = pd.DataFrame({"gene": genes, "score": score, "p-value": pvalue})
    return final
