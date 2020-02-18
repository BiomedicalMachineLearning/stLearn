from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import matplotlib
import numpy as np
import networkx as nx
from stlearn._compat import Literal
from typing import Optional, Union
from anndata import AnnData
import warnings
#from .utils import get_img_from_fig, checkType
from .extra import bfs_edges_modified
from stlearn.plotting.trajectory.utils import checkType
from scipy.stats import spearmanr
from statsmodels.sandbox.stats.multicomp import multipletests

from copy import deepcopy


def transition_genes(
    adata: AnnData,
    subcluster: int = 0,
    cutoff_spearman: float = 0.4,
    cutoff_logfc: float = 0.25,
    percentile_expr: int = 95,
    n_jobs: int = 1,
    min_num_cells: int = 5,
    use_precomputed: bool  = True,
    root: str = 'S0',
    copy: bool = False,
) -> Optional[AnnData]:
    """Detect transition genes along one branch.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    cutoff_spearman: `float`, optional (default: 0.4)
        Between 0 and 1. The cutoff used for Spearman's rank correlation coefficient.
    cutoff_logfc: `float`, optional (default: 0.25)
        The log-transformed fold change cutoff between cells around start and end node.
    percentile_expr: `int`, optional (default: 95)
        Between 0 and 100. Between 0 and 100. Specify the percentile of gene expression greater than 0 to filter out some extreme gene expressions. 
    min_num_cells: `int`, optional (default: 5)
        The minimum number of cells in which genes are expressed.
    n_jobs: `int`, optional (default: 1)
        The number of parallel jobs to run when scaling the gene expressions .
    use_precomputed: `bool`, optional (default: True)
        If True, the previously computed scaled gene expression will be used
    root: `str`, optional (default: 'S0'): 
        The starting node
    preference: `list`, optional (default: None): 
        The preference of nodes. The branch with speficied nodes are preferred and will be dealt with first. The higher ranks the node have, The earlier the branch with that node will be analyzed. 
        This will help generate the consistent results as shown in subway map and stream plot.
    Returns
    -------
    updates `adata` with the following fields.
    scaled_gene_expr: `list` (`adata.uns['scaled_gene_expr']`)
        Scaled gene expression for marker gene detection.    
    transition_genes: `dict` (`adata.uns['transition_genes']`)
        Transition genes for each branch deteced by STREAM.
    """

    subcl_adata = adata.uns["subcluster_" + str(subcluster) + "_adata"]

    df_gene_detection = subcl_adata.obs.copy()
    df_gene_detection.rename(columns={"branch_lam": "lam"},inplace = True)

    flat_tree = subcl_adata.uns["pseudotimespace"]['flat_tree']
    dict_node_state = nx.get_node_attributes(flat_tree,'label')

    df_sc = pd.DataFrame(index= subcl_adata.obs_names.tolist(),
                             data = subcl_adata.obsm["filtered_counts"],
                             columns=subcl_adata.var_names.tolist())

    input_genes = subcl_adata.var_names.tolist()

    print("Filtering out genes that are expressed in less than " + str(min_num_cells) + " cells ...")

    input_genes_expressed = np.array(input_genes)[np.where((df_sc[input_genes]>0).sum(axis=0)>min_num_cells)[0]].tolist()

    print(str(len(input_genes_expressed)) + ' genes are being scanned ...')
    
    df_gene_detection[input_genes_expressed] = df_sc[input_genes_expressed].copy()

    df_scaled_gene_expr = df_gene_detection

    dict_tg_edges = dict()
    dict_label_node = {value: key for key,value in nx.get_node_attributes(flat_tree,'label').items()}

    root_node = dict_label_node[root]

    bfs_edges = bfs_edges_modified(flat_tree,root_node,preference=None)
    current_pseudotime = subcl_adata.obs[["node",root + "_pseudotime"]]

    branch_i_nodes = bfs_edges[0]

    direction_arr = []
    for node in branch_i_nodes:
        tmpx = current_pseudotime[current_pseudotime["node"]==node]
        if len(tmpx)>0:
            direction_arr.append(tmpx.iloc[:,1][0])
            
    if  len(direction_arr) > 1:
        if not checkType(direction_arr):
            branch_i_nodes = branch_i_nodes[::-1]


    from copy import deepcopy
    for edge_i in bfs_edges:
        
        direction_arr = []
        for node in edge_i:
            tmpx = current_pseudotime[current_pseudotime["node"]==node]
            if len(tmpx)>0:
                direction_arr.append(tmpx.iloc[:,1][0])

        if  len(direction_arr) > 1:
            if not checkType(direction_arr):
                edge_i = edge_i[::-1]
            
        edge_i_alias = (dict_node_state[edge_i[0]],dict_node_state[edge_i[1]])
        print("Processing:")
        print(edge_i_alias)
        if edge_i in nx.get_edge_attributes(flat_tree,'id').values():
            df_cells_edge_i = deepcopy(df_gene_detection[df_gene_detection.branch_id==edge_i])
            df_cells_edge_i['lam_ordered'] = df_cells_edge_i['lam']
        else:
            df_cells_edge_i = deepcopy(df_gene_detection[df_gene_detection.branch_id==(edge_i[1],edge_i[0])])
            df_cells_edge_i['lam_ordered'] = flat_tree.edges[edge_i]['len'] - df_cells_edge_i['lam']
        df_cells_edge_i_sort = df_cells_edge_i.sort_values(['lam_ordered'])
        df_stat_pval_qval = pd.DataFrame(columns = ['stat','logfc','pval','qval'],dtype=float)
        for genename in input_genes_expressed:
            id_initial = range(0,int(df_cells_edge_i_sort.shape[0]*0.2))
            id_final = range(int(df_cells_edge_i_sort.shape[0]*0.8),int(df_cells_edge_i_sort.shape[0]*1))
            values_initial = df_cells_edge_i_sort.iloc[id_initial,:][genename]
            values_final = df_cells_edge_i_sort.iloc[id_final,:][genename]
            diff_initial_final = abs(values_final.mean() - values_initial.mean())
            if(diff_initial_final>0):
                logfc = np.log2(max(values_final.mean(),values_initial.mean())/(min(values_final.mean(),values_initial.mean())+diff_initial_final/1000.0))
            else:
                logfc = 0
            if(logfc>cutoff_logfc):
                df_stat_pval_qval.loc[genename] = np.nan
                df_stat_pval_qval.loc[genename,['stat','pval']] = spearmanr(df_cells_edge_i_sort.loc[:,genename],\
                                                                            df_cells_edge_i_sort.loc[:,'lam_ordered'])
                df_stat_pval_qval.loc[genename,'logfc'] = logfc
        if(df_stat_pval_qval.shape[0]==0):
            print('No Transition genes are detected in branch ' + edge_i_alias[0]+'_'+edge_i_alias[1])
        else:
            p_values = df_stat_pval_qval['pval']
            q_values = multipletests(p_values, method='fdr_bh')[1]
            df_stat_pval_qval['qval'] = q_values
            dict_tg_edges[edge_i_alias] = df_stat_pval_qval[(abs(df_stat_pval_qval.stat)>=cutoff_spearman)].sort_values(['qval'])
    subcl_adata.uns['transition_genes'] = dict_tg_edges  