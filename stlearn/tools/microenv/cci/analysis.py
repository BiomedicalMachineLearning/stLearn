""" Wrapper function for performing CCI analysis, varrying the analysis based on
    the inputted data / state of the anndata object.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from numba.typed import List

from anndata import AnnData
from sklearn.cluster import AgglomerativeClustering
from .base import calc_neighbours, get_lrs_scores, calc_distance
from .base_grouping import get_hotspots
from .het import count, count_interactions, get_interactions, \
                                                           get_data_for_counting
from .permutation import perform_perm_testing, perform_spot_testing

def run(adata: AnnData, lrs: np.array,
        use_label: str = None, use_het: str = 'cci_het',
        distance: int = 0, n_pairs: int = 1000, neg_binom: bool = False,
        adj_method: str = 'fdr_bh', pval_adj_cutoff: float = 0.05,
        lr_mid_dist: int = 150, min_spots: int = 10, min_expr: float = 0,
        verbose: bool = True, method: str='spot_sig', quantile=0.05,
        plot_diagnostics: bool = False, show_plot=False, save_bg=False,
        n_groups: int=5,
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
    n_groups:               If in spot_sig mode, then groups LRs into groups of 10, then generate background for each.
    method: str             One of spot_sig, lr_sig, or hotspot; former generates background for each spot, middle generates bg for each lr, and latter identifies hotspots. 'lr_sig'/'hotspot' will be removed in future pushes.
    Returns
    -------
    adata: AnnData          Relevant information stored: adata.uns['het'], adata.uns['lr_summary'], & data.uns['per_lr_results'].
    """
    distance = calc_distance(adata, distance)
    neighbours = calc_neighbours(adata, distance, verbose=verbose)
    adata.obsm['spot_neighbours'] = pd.DataFrame([','.join(x.astype(str))
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

    if method == 'spot_sig':
        """ Permutation methods generating background per spot, & testing lrs in spot. 
        """
        perform_spot_testing(adata, lr_scores, lrs, n_pairs, neighbours,
                             het_vals, min_expr, adj_method, pval_adj_cutoff,
                                                                        verbose,
                             save_bg=save_bg, n_groups=n_groups)

    elif method == 'lr_sig':
        """ Permutation based method generating backgrounds per lr/lr group.
          1. Group LRs with similar mean expression.
          2. Calc. common bg distrib. for grouped lrs.
          3. Calc. p-values for each lr relative to bg. 
        """
        perform_perm_testing(adata, lr_scores, n_pairs, lrs, lr_mid_dist,
                             verbose, neighbours, het_vals, min_expr,
                             neg_binom, adj_method, pval_adj_cutoff,
                            )
    else:
        """ Perform per lr background removal to get hot-spots by choosing dynamic cutoffs.
        Inspired by the watershed method:
            1. Generate set of candidate cutoffs based on quantiles.
            2. Perform DBScan clustering using spatial coordinates at different cutoffs.
            3. Choose cutoff as point that maximises number of clusters i.e. peaks. 
        """
        # TODO need to evaluate this eps param better.
        eps = 3*distance if type(distance)!=type(None) and distance!=0 else 100
        get_hotspots(adata, lr_scores.transpose(), lrs, eps=eps,
                     quantile=quantile, verbose=verbose,
                     plot_diagnostics=plot_diagnostics, show_plot=show_plot)


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

################################################################################
            # Functions for calling Celltype-Celltype interactions #
################################################################################
def run_cci(adata: AnnData, use_label: str,
            min_spots: int = 3, sig_spots: bool = True,
            spot_mixtures: bool = False, cell_prop_cutoff: float = 0.2,
            verbose: bool = True,
            ):
    ran_lr = 'lr_summary' in adata.uns
    ran_sig = False if not ran_lr else 'n_spots_sig' in adata.uns['lr_summary'].columns
    if not ran_lr and not ran_sig:
        raise Exception("No LR results testing results found, " 
                        "please run st.tl.cci.run first")

    # Ensuring compatibility with current way of adding label_transfer to object
    if use_label == "label_transfer" or use_label == "predictions":
        obs_key, uns_key = "predictions", "label_transfer"
    else:
        obs_key, uns_key = use_label, use_label

    # Determining the neighbour spots used for significance testing #
    neighbours = List()
    for i in range(adata.obsm['spot_neighbours'].shape[0]):
        neighs = np.array(adata.obsm['spot_neighbours'].values[i,
                                                               :][0].split(','))
        neighs = neighs[neighs != ''].astype(int)
        neighs = neighs[neighs<adata.shape[0]] # Removing subsetted spots..
        neighbours.append(neighs)

    # Getting the cell/tissue types that we are actually testing #
    tissue_types = adata.obs[obs_key].values.astype(str)
    all_set = np.unique(tissue_types)

    # Mixture mode
    mix_mode = spot_mixtures and uns_key in adata.uns
    if not mix_mode and spot_mixtures:
        print(f"Warning: specified spot_mixtures but no deconvolution data in "
              f"adata.uns['{uns_key}'].\nFalling back to discrete mode.")

    # Getting minimum necessary information for edge counting #
    spot_bcs, cell_data, neighbourhood_bcs, neighbourhood_indices = \
                            get_data_for_counting(adata, use_label,
                                                  mix_mode, neighbours, all_set)


    lr_summary = adata.uns['lr_summary']
    col_i = 1 if sig_spots else 0
    col = 'lr_sig_scores'if sig_spots else 'lr_scores'
    best_lrs = lr_summary.index.values[lr_summary.values[:,col_i] > min_spots]
    all_matrix = np.zeros((len(all_set), len(all_set)), dtype=int)
    per_lr_cci = {}
    for best_lr in best_lrs:
        l, r = best_lr.split('_')
        #lr_results = adata.uns['per_lr_results'][best_lr]

        L_bool = adata[:, l].X.toarray()[:, 0] > 0
        R_bool = adata[:, r].X.toarray()[:, 0] > 0
        lr_index = np.where(adata.uns['lr_summary'].index.values==best_lr)[0][0]
        sig_bool = adata.obsm[col][:, lr_index] > 0
        #sig_bool = lr_results.loc[:, 'lr_sig_scores'].values != 0

        # Now counting the interactions under 3 situations:
        # 1) sig spot with ligand, only neighbours with receptor relevant
        # 2) sig spot with receptor, only neighbours with ligand relevant
        # NOTE, A<->B is double counted, but on different side of matrix.
        # (if bidirectional interaction between two spots, counts as two seperate interactions).
        LR_edges = get_interactions(spot_bcs, cell_data,
                                    neighbourhood_bcs, neighbourhood_indices,
                                    all_set, mix_mode, sig_bool,
                                    L_bool, R_bool,
                                    tissue_types=tissue_types,
                                    cell_prop_cutoff=cell_prop_cutoff,
                                    trans_dir=True, #sig ligand->receptor mode
                                    )
        RL_edges = get_interactions(spot_bcs, cell_data,
                                    neighbourhood_bcs, neighbourhood_indices,
                                    all_set, mix_mode, sig_bool,
                                    R_bool, L_bool,
                                    tissue_types=tissue_types,
                                    cell_prop_cutoff=cell_prop_cutoff,
                                    trans_dir=False, #sig receptor->ligand mode
                                    )

        # Counting the number of unique interacting edges
        # between different cell type via indicate LR
        int_matrix = np.zeros((len(all_set), len(all_set)), dtype=int)
        for i, cell_A in enumerate(all_set):
            RL_Atrans_edges = RL_edges[cell_A]
            LR_Atrans_edges = LR_edges[cell_A]
            for j, cell_B in enumerate(all_set):
                RL_Atrans_Bedges = RL_Atrans_edges[cell_B]
                LR_Atrans_Bedges = LR_Atrans_edges[cell_B]
                Atrans_Bedges = np.unique( RL_Atrans_Bedges+LR_Atrans_Bedges )
                int_matrix[i, j] = len(Atrans_Bedges)

        all_matrix += int_matrix
        int_matrix = pd.DataFrame(int_matrix, index=all_set, columns=all_set)
        per_lr_cci[best_lr] = int_matrix

    adata.uns[f'lr_cci_{use_label}'] = pd.DataFrame(all_matrix,
                                             index=all_set, columns=all_set)
    adata.uns[f'per_lr_cci_{use_label}'] = per_lr_cci
    if verbose:
        print(f"Counts of cci interactions for all LR pairs in "
              f"{f'lr_cci_{use_label}'}")
        print(f"Counts of cci interactions for each LR pair stored in dictionary"
              f" {f'per_lr_cci_{use_label}'}")








