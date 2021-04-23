""" Wrapper function for performing CCI analysis, varrying the analysis based on
    the inputted data / state of the anndata object.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from numba import njit, prange
from anndata import AnnData
from sklearn.cluster import AgglomerativeClustering
from .base import calc_neighbours, get_spot_lrs, calc_distance
from .het import count
from .permutation import get_scores, get_stats, get_valid_genes, get_ordered, \
                                                get_median_index, get_rand_pairs

def run(adata: AnnData, lrs: np.array,
        use_label: str = None, use_het: str = 'cci_het',
        distance: int = 0, n_pairs: int = 1000, neg_binom: bool = False,
        adj_method: str = 'fdr_bh', pval_adj_cutoff: float = 0.01,
        lr_mid_dist: int = 150, min_spots: int = 5, min_expr: float = 0.5,
        verbose: bool = True,
        ):
    """Wrapper function for performing CCI analysis, varrying the analysis based 
        on the inputted data / state of the anndata object.
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count.
    lrs:    np.array        The LR pairs to score/test for enrichment (in format 'L1_R1')
    use_label: str          The cell type results to use in counting.
    use_het:                The storage place for cell heterogeneity results in adata.obsm.
    distance: int           Distance to determine the neighbours (default is the nearest neighbour), distance=0 means within spot
    n_pairs: int            Number of random pairs to generate when performing the background distribution.
    neg_binom: bool         Whether to use neg-binomial distribution to estimate p-values, NOT appropriate with log1p data, alternative is to use background distribution itself (recommend higher number of n_pairs for this).
    adj_method: str         Parsed to statsmodels.stats.multitest.multipletests for multiple hypothesis testing correction.
    lr_mid_dist: int        The distance between the mid-points of the average expression of the two genes in an LR pair for it to be group with other pairs via AgglomerativeClustering to generate a common background distribution.
    min_spots: int          Minimum number of spots with an LR score to be considered for further testing.
    min_expr: float         Minimum gene expression of either L or R for spot to be considered to have reasonable score.
    Returns
    -------
    adata: AnnData          Relevant information stored: adata.uns['het'], adata.uns['lr_summary'], & data.uns['per_lr_results'].
    """
    distance = calc_distance(adata, distance)
    neighbours = calc_neighbours(adata, distance, verbose=verbose)
    adata.uns['spot_neighbours'] = pd.DataFrame([','.join(x.astype(str))
                                                           for x in neighbours],
                           index=adata.obs_names, columns=['neighbour_indices'])
    if verbose:
        print("Spot neighbour indices stored in adata.uns['spot_neighbours']")

    # Conduct with cell heterogeneity info if label_transfer provided #
    cell_het = type(use_label) != type(None) and use_label in adata.uns.keys()
    if cell_het:
        if verbose:
            print("Calculating cell hetereogeneity...")

        # Calculating cell heterogeneity #
        count(adata, distance=distance, use_label=use_label, use_het=use_het)

    het_vals = np.array([1] * len(adata)) \
                           if use_het not in adata.obsm else adata.obsm[use_het]

    """ 1. Filter any LRs without stored expression.
          2. Group LRs with similar mean expression.
          3. Calc. common bg distrib. for grouped lrs.
          4. Calc. p-values for each lr relative to bg. 
    """
    # Calculating the lr_scores across spots for the inputted lrs #
    lr_scores, lrs = get_lrs_scores(adata, lrs, neighbours, het_vals, min_expr)
    lr_bool = (lr_scores>0).sum(axis=0) > min_spots
    lrs = lrs[lr_bool]
    lr_scores = lr_scores[:, lr_bool]
    if verbose:
        print("Altogether " + str(len(lrs)) + " valid L-R pairs")
    if len(lrs) == 0:
        print("Exiting due to lack of valid LR pairs.")
        return

    if n_pairs != 0: #Perform permutation testing
        # Grouping spots with similar mean expression point #
        genes = get_valid_genes(adata, n_pairs)
        means_ordered, genes_ordered = get_ordered(adata, genes)
        ims = np.array(
                     [get_median_index(lr_.split('_')[0], lr_.split('_')[1],
                                        means_ordered.values, genes_ordered)
                        for lr_ in lrs]).reshape(-1, 1)

        if len(lrs) > 1: # Multi-LR pair mode, group LRs to generate backgrounds
            clusterer = AgglomerativeClustering(n_clusters=None,
                                                distance_threshold=lr_mid_dist,
                                                affinity='manhattan',
                                                linkage='single')
            lr_groups = clusterer.fit_predict(ims)
            lr_group_set = np.unique(lr_groups)
            if verbose:
                print(f'{len(lr_group_set)} lr groups with similar expression levels.')

        else: #Single LR pair mode, generate background for the LR.
            lr_groups = np.array([0])
            lr_group_set = lr_groups

        res_info = ['lr_scores', 'p_val', 'p_adj', '-log10(p_adj)',
                                                                'lr_sig_scores']
        n_, n_sigs = np.array([0]*len(lrs)), np.array([0]*len(lrs))
        per_lr_results = {}
        with tqdm(
                total=len(lr_group_set),
                desc="Generating background distributions for the LR pair groups..",
                bar_format="{l_bar}{bar} [ time left: {remaining} ]",
        ) as pbar:
            for group in lr_group_set:
                # Determining common mid-point for each group #
                group_bool = lr_groups==group
                group_im = int(np.median(ims[group_bool, 0]))

                # Calculating the background #
                rand_pairs = get_rand_pairs(adata, genes, n_pairs,
                                                           lrs=lrs, im=group_im)
                background = get_lrs_scores(adata, rand_pairs, neighbours,
                                            het_vals, min_expr,
                                                     filter_pairs=False).ravel()
                total_bg = len(background)
                background = background[background!=0] #Filtering for increase speed

                # Getting stats for each lr in group #
                group_lr_indices = np.where(group_bool)[0]
                for lr_i in group_lr_indices:
                    lr_ = lrs[lr_i]
                    lr_results = pd.DataFrame(index=adata.obs_names,
                                                               columns=res_info)
                    scores = lr_scores[:, lr_i]
                    stats = get_stats(scores, background, total_bg, neg_binom,
                                    adj_method, pval_adj_cutoff=pval_adj_cutoff)
                    full_stats = [scores]+list(stats)
                    for vals, colname in zip(full_stats, res_info):
                        lr_results[colname] = vals

                    n_[lr_i] = len(np.where(scores>0)[0])
                    n_sigs[lr_i] = len(np.where(
                                            lr_results['p_adj'].values<0.05)[0])
                    if n_sigs[lr_i] > 0:
                        per_lr_results[lr_] = lr_results
                pbar.update(1)

        print(f"{len(per_lr_results)} LR pairs with significant interactions.")

        lr_summary = pd.DataFrame(index=lrs, columns=['n_spots', 'n_spots_sig'])
        lr_summary['n_spots'] = n_
        lr_summary['n_spots_sig'] = n_sigs
        lr_summary = lr_summary.iloc[np.argsort(-n_sigs)]

    else: #Simply stored the scores
        per_lr_results = {}
        lr_summary = pd.DataFrame(index=lrs, columns=['n_spots'])
        for i, lr_ in enumerate(lrs):
            lr_results = pd.DataFrame(index=adata.obs_names,
                                                          columns=['lr_scores'])
            lr_results['lr_scores'] = lr_scores[:, i]
            per_lr_results[lr_] = lr_results
            lr_summary.loc[lr_, 0] = len(np.where(lr_scores[:, i]>0)[0])
        lr_summary = lr_summary.iloc[np.argsort(-lr_summary.values[:,0]),:]

    adata.uns['lr_summary'] = lr_summary
    adata.uns['per_lr_results'] = per_lr_results
    if verbose:
        print("Summary of significant spots for each lr pair in adata.uns['lr_summary'].")
        print("Spot enrichment statistics of LR interactions in adata.uns['per_lr_results']")

def get_lrs_scores(adata: AnnData, lrs: np.array, neighbours: np.array,
                   het_vals: np.array, min_expr: float,
                   filter_pairs: bool = True
                   ):
    """Gets the scores for the indicated set of LR pairs & the heterogeneity values.
    Parameters
    ----------
    adata: AnnData   See run() doc-string.
    lrs: np.array    See run() doc-string.
    neighbours: np.array    Array of arrays with indices specifying neighbours of each spot.
    het_vals: np.array      Cell heterogeneity counts per spot.
    min_expr: float         Minimum gene expression of either L or R for spot to be considered to have reasonable score.
    filter_pairs: bool      Whether to filter to valid pairs or not.
    Returns
    -------
    lrs: np.array   lr pairs from the database in format ['L1_R1', 'LN_RN']
    """
    spot_lr1s = get_spot_lrs(adata, lr_pairs=lrs, lr_order=True,
                                                      filter_pairs=filter_pairs)
    spot_lr2s = get_spot_lrs(adata, lr_pairs=lrs, lr_order=False,
                                                      filter_pairs=filter_pairs)
    if filter_pairs:
        lrs = np.array(['_'.join(spot_lr1s.columns.values[i:i + 2])
                        for i in range(0, spot_lr1s.shape[1], 2)])

    # Calculating the expression filter to make sure low expression spots filtered
    ls = np.array([lr.split('_')[0] for lr in lrs])
    rs = np.array([lr.split('_')[1] for lr in lrs])
    lrs_ = np.array(list(ls)+list(rs))
    lr_df = adata.to_df().loc[:, lrs_]
    lr_indices = np.array([
                            [np.where(lr_df.columns.values==ls[i])[0][0],
                             np.where(lr_df.columns.values==rs[i])[0][0]]
                                                       for i in range(len(ls))])
    expr_filter = calc_expr_filter(lr_df.values, lr_indices, min_expr)

    # Calculating the lr_scores across spots for the inputted lrs #
    lr_scores = get_scores(spot_lr1s.values, spot_lr2s.values,
                           neighbours, het_vals, expr_filter)

    if filter_pairs:
        return lr_scores, lrs
    else:
        return lr_scores

@njit(parallel=True)
def calc_expr_filter(lr_expr: np.array, lr_indices: np.array, min_expr: float):
    # Determining spots to filter scores if insignificantly express L or R #
    expr_filter = np.zeros((lr_expr.shape[0], lr_indices.shape[0]),
                                                                  dtype=np.int_)
    for j in prange(expr_filter.shape[1]):
        l_bool = lr_expr[:,lr_indices[j,0]] > min_expr
        r_bool = lr_expr[:,lr_indices[j,1]] > min_expr
        for i in range(expr_filter.shape[0]):
            if l_bool[i] or r_bool[i]:
                expr_filter[i, j] = 1
            else:
                expr_filter[i, j] = 0
    return expr_filter

def load_lrs(names: str) -> np.array:
    """Loads inputted LR database, & concatenates into consistent database set of pairs without duplicates.
    Parameters
    ----------
    names: str   Databases to load, options: \
                'connectomeDB2020_lit' (literature verified), 'connectomeDB2020_put' (putative). \
                If more than one specified, loads all & removes duplicates.
    Returns
    -------
    lrs: np.array   lr pairs from the database in format ['L1_R1', 'LN_RN']
    """
    path = os.path.dirname(os.path.realpath(__file__))
    dbs = [pd.read_csv(f'{path}/databases/{name}.txt', sep='\t')
                                                              for name in names]
    lrs_full = []
    for db in dbs:
        lrs = [f'{db.values[i,0]}_{db.values[i,1]}' for i in range(db.shape[0])]
        lrs_full.extend(lrs)
    return np.unique(lrs_full)






