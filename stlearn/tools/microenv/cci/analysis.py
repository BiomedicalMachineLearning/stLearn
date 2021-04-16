""" Wrapper function for performing CCI analysis, varrying the analysis based on
    the inputted data / state of the anndata object.
"""

import os
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from .base import lr, calc_neighbours, get_spot_lrs
from .het import count
from .merge import merge
from .permutation import permutation, get_scores, get_stats, \
                  get_valid_genes, get_ordered, get_median_index, get_rand_pairs

def run(adata, lrs, use_label=None, distance=0, n_pairs=0, verbose=True,
        neg_binom: bool = False, adj_method: str = 'fdr', run_fast=True,
        bg_pairs = None, min_spots=5,
        **kwargs):
    """Wrapper function for performing CCI analysis, varrying the analysis based 
        on the inputted data / state of the anndata object.
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count.
    use_label:              The cell type results to use in counting.
    use_het:                The storage place for cell heterogeneity results.
    distance: int           Distance to determine the neighbours (default is the nearest neighbour), distance=0 means within spot
    **kwargs:               Extra arguments parsed to permutation.
    Returns
    -------
    adata: AnnData          With the counts of specified clusters in nearby spots stored as adata.uns['het']
    """
    adata.uns['lr'] = lrs
    neighbours = calc_neighbours(adata, distance,
                         index=True if 'fast' not in kwargs else kwargs['fast'])

    # Conduct with cell heterogeneity info if label_transfer provided #
    cell_het = type(use_label) != type(None) and use_label in adata.uns.keys()
    use_het = 'cci_het' if cell_het else None
    if cell_het:
        if verbose:
            print("Calculating cell hetereogeneity & merging with LR scores...")

        # Calculating cell heterogeneity #
        count(adata, distance=distance, use_label=use_label)

    if len(lrs) == 1: #Single LR mode
        lr(adata=adata, distance=distance, neighbours=neighbours, **kwargs)
        if cell_het: # Merging with the lR values #
            merge(adata, use_lr='cci_lr', use_het='cci_het')

        if n_pairs != 0:  # Permutation testing #
            print("Performing permutation testing...")
            res = permutation(adata, use_het=use_het, n_pairs=n_pairs,
                              distance=distance, neg_binom=neg_binom,
                              adj_method=adj_method, neighbours=neighbours,
                              run_fast=run_fast, bg_pairs=bg_pairs,
                              **kwargs)
            return res

    else: #Multi-LR mode
        """ 1. Filter any LRs without stored expression.
            2. Group LRs with similar mean expression.
            3. Calc. common bg distrib. for grouped lrs.
            4. Calc. p-values for each lr relative to bg. 
        """
        het_vals = np.array([1] * len(adata)) if use_het == None else \
                                                             adata.obsm[use_het]

        # Calculating the lr_scores across spots for the inputted lrs #
        lr_scores, lrs = get_lrs_scores(adata, lrs, neighbours, het_vals)
        lr_bool = (lr_scores>0).sum(axis=0) > min_spots
        lrs = lrs[lr_bool]
        lr_scores = lr_scores[:, lr_bool]
        if verbose:
            print("Altogether " + str(len(lrs)) + " valid L-R pairs")

        if n_pairs != 0:
            # Grouping spots with similar mean expression point #
            genes = get_valid_genes(adata, n_pairs)
            means_ordered, genes_ordered = get_ordered(adata, genes)
            ims = np.array(
                         [get_median_index(lr_.split('_')[0], lr_.split('_')[1],
                                            means_ordered.values, genes_ordered)
                            for lr_ in lrs]).reshape(-1, 1)
            clusterer = AgglomerativeClustering(n_clusters=None,
                                                distance_threshold=n_pairs,
                                                affinity='manhattan',
                                                linkage='single')
            lr_groups = clusterer.fit_predict(ims)
            lr_group_set = np.unique(lr_groups)
            print(f'{len(lr_group_set)} lr groups with similar expression levels.')
            print('Generating background for each group, may take a while...')
            res_info = ['lr_scores', 'p_val', 'p_adj', '-log10(p_adj)',
                                                                'lr_sig_scores']
            n_sigs = np.array([0]*len(lrs))
            per_lr_results = {}
            for group in lr_group_set:
                # Determining common mid-point for each group #
                group_bool = lr_groups==group
                group_im = int(np.median(ims[group_bool, 0]))

                # Calculating the background #
                rand_pairs = get_rand_pairs(adata, genes, n_pairs,
                                                           lrs=lrs, im=group_im)
                background = get_lrs_scores(adata, rand_pairs, neighbours,
                                           het_vals, filter_pairs=False).ravel()

                # Getting stats for each lr in group #
                group_lr_indices = np.where(group_bool)[0]
                for lr_i in group_lr_indices:
                    lr_ = lrs[lr_i]
                    lr_results = pd.DataFrame(index=adata.obs_names,
                                                               columns=res_info)
                    scores = lr_scores[:, lr_i]
                    stats = get_stats(scores, background, neg_binom, adj_method)
                    full_stats = [scores]+list(stats)
                    for vals, colname in zip(full_stats, res_info):
                        lr_results[colname] = vals

                    n_sigs[lr_i] = len(np.where(lr_results['p_adj'].values<0.05)[0])
                    if n_sigs[lr_i] > 1:
                        per_lr_results[lr_] = lr_results

            lr_summary = pd.DataFrame(index=lrs, columns=['n_spots_sig'])
            lr_summary['n_spots_sig'] = n_sigs

            print(f"{len(per_lr_results)} LR pairs with significant interactions.")

            return lr_summary, per_lr_results

def get_lrs_scores(adata, lrs, neighbours, het_vals, filter_pairs=True):
    spot_lr1s = get_spot_lrs(adata, lr_pairs=lrs, lr_order=True,
                                                      filter_pairs=filter_pairs)
    spot_lr2s = get_spot_lrs(adata, lr_pairs=lrs, lr_order=False,
                                                      filter_pairs=filter_pairs)

    # Calculating the lr_scores across spots for the inputted lrs #
    lr_scores = get_scores(spot_lr1s.values, spot_lr2s.values,
                           neighbours, het_vals)
    if filter_pairs:
        lrs = np.array(['_'.join(spot_lr1s.columns.values[i:i + 2])
                        for i in range(0, spot_lr1s.shape[1], 2)])
        return lr_scores, lrs
    else:
        return lr_scores

def load_lrs(names: str) -> np.array:
    """
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






