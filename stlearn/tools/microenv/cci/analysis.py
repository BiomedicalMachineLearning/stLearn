""" Wrapper function for performing CCI analysis, varrying the analysis based on
    the inputted data / state of the anndata object.
"""

import os
import numba
import numpy as np
import pandas as pd
from typing import Union
from anndata import AnnData
from .base import calc_neighbours, get_lrs_scores, calc_distance
from .permutation import perform_spot_testing
from .go import run_GO
from .het import count, get_data_for_counting, get_interaction_matrix, \
                         get_interaction_pvals

################################################################################
            # Functions related to Ligand-Receptor interactions #
################################################################################
def load_lrs(names: Union[str, list, None]=None, species: str='human') \
                                                                    -> np.array:
    """Loads inputted LR database, & concatenates into consistent database set of pairs without duplicates. If None loads 'connectomeDB2020_lit'.
    Parameters
    ----------
    names: list   Databases to load, options: \
                'connectomeDB2020_lit' (literature verified), 'connectomeDB2020_put' (putative). \
                If more than one specified, loads all & removes duplicates.
    species: str    Format of the LR genes, either 'human' or 'mouse'.

    Returns
    -------
    lrs: np.array   lr pairs from the database in format ['L1_R1', 'LN_RN']
    """
    if type(names)==type(None):
        names = ['connectomeDB2020_lit']
    if type(names)==str:
        names = [names]

    path = os.path.dirname(os.path.realpath(__file__))
    dbs = [pd.read_csv(f'{path}/databases/{name}.txt', sep='\t')
                                                              for name in names]
    lrs_full = []
    for db in dbs:
        lrs = [f'{db.values[i,0]}_{db.values[i,1]}' for i in range(db.shape[0])]
        lrs_full.extend(lrs)
    lrs_full = np.unique(lrs_full)
    # If dealing with mouse, need to reformat #
    if species == 'mouse':
        genes1 = [lr_.split('_')[0] for lr_ in lrs_full]
        genes2 = [lr_.split('_')[1] for lr_ in lrs_full]
        lrs_full = np.array([genes1[i][0]+genes1[i][1:].lower()+'_'+\
                             genes2[i][0]+genes2[i][1:].lower()
                             for i in range(len(lrs_full))])

    return lrs_full

def run(adata: AnnData, lrs: np.array,
        use_label: str = None, use_het: str = 'cci_het',
        distance: int = None, n_pairs: int = 1000,
        adj_method: str = 'fdr_bh', pval_adj_cutoff: float = 0.05,
        min_spots: int = 10, min_expr: float = 0,
        save_bg: bool=False, n_cpus: int=None, verbose: bool = True,
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
    adj_method: str         Parsed to statsmodels.stats.multitest.multipletests for multiple hypothesis testing correction.
    pval_adj_cutoff: float  P-value below which LR is considered significant in spot neighbourhood.
    pval_adj_axis: str      Defines how pval adjustment performed; 'lrs_per_spot' to correct by no. of LRs tested in spot, 'spots_per_lr' to correct for no. of spots tested for LR.
    min_spots: int          Minimum number of spots with an LR score to be considered for further testing.
    min_expr: float         Minimum gene expression of either L or R for spot to be considered to have reasonable score.
    save_bg: bool           Whether to save the background per LR pair; for method development only. Not recommended since huge memory.
    n_cpus: int             The number of cpus to use for multi-threading; by default will use all available.
    verbose: bool           True if print dialogue to user during run-time.
    Returns
    -------
    adata: AnnData          Relevant information stored: adata.uns['het'], adata.uns['lr_summary'], & data.uns['per_lr_results'].
    """
    # Setting threads for paralellisation #
    if type(n_cpus)!=type(None):
        numba.set_num_threads(n_cpus)

    # Calculating neighbour & storing #
    distance = calc_distance(adata, distance)
    neighbours = calc_neighbours(adata, distance, verbose=verbose)
    adata.obsm['spot_neighbours'] = pd.DataFrame([','.join(x.astype(str))
                                                           for x in neighbours],
                           index=adata.obs_names, columns=['neighbour_indices'])
    spot_neighs_df = adata.obsm['spot_neighbours']
    spot_neigh_bcs = []
    for i in range(spot_neighs_df.shape[0]):
        neigh_indices = [int(index) for index in
                         spot_neighs_df.values[i, 0].split(',') if index!='']
        neigh_bcs = [adata.obs_names[index] for index in neigh_indices]
        spot_neigh_bcs.append(','.join(neigh_bcs))
    spot_neigh_bcs_df = pd.DataFrame(spot_neigh_bcs, index=spot_neighs_df.index,
                                     columns=['neighbour_bcs'])
    # Important to store barcodes in-case adata subsetted #
    adata.obsm['spot_neigh_bcs'] = spot_neigh_bcs_df

    if verbose:
        print("Spot neighbour indices stored in adata.obsm['spot_neighbours'] "
              "& adata.obsm['spot_neigh_bcs'].")

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

    """ Permutation methods generating background per spot, & test lrs in spot. 
    """
    perform_spot_testing(adata, lr_scores, lrs, n_pairs, neighbours,
                         het_vals, min_expr, adj_method, pval_adj_cutoff,
                                                                    verbose,
                                                                save_bg=save_bg)

def run_lr_go(adata: AnnData, r_path: str, n_top: int=100,
              bg_genes: np.array=None,
              min_sig_spots: int=1, species: str='human',
              p_cutoff: float=.01, q_cutoff: float=0.5, onts: str='BP',
              verbose: bool=True):
    """ Runs a basic GO analysis on the genes in the top ranked LR pairs.
        Only supported for human and mouse species.

    Parameters
    ----------
    adata: AnnData          Must have had st.tl.cci_rank.run() called prior.
    r_path: str             Path to R, must have clusterProfiler installed.
    bg_genes: np.array      Genes to be used as the background. If None, defaults to all genes in lr database: 'connectomeDB2020_put'.
    n_top: int              The top number of LR pairs to use.
    min_sig_spots: int      Minimum no. of significant spots pairs must have to be considered.
    species: str            Species to perform the GO testing for.
    p_cutoff: float         P-value & P-adj cutoff below which results will be returned.
    q_cutoff: float         Q-value cutoff below which results will be returned.
    onts: str               As per clusterProfiler; One of "BP", "MF", and "CC" subontologies, or "ALL" for all three.
    Returns
    -------
    adata: AnnData          Relevant information stored: adata.uns['lr_go']
    """
    #### Making sure inputted correct species ####
    all_species = ['human', 'mouse']
    if species not in all_species:
        raise Exception(f'Got {species} for species, must be one of '
                        f'{all_species}')

    #### Getting the genes from the top LR pairs ####
    if 'lr_summary' not in adata.uns:
        raise Exception("Need to run st.tl.cci_rank.run first.")
    lrs = adata.uns['lr_summary'].index.values.astype(str)
    n_sig = adata.uns['lr_summary'].loc[:,'n_spots_sig'].values.astype(int)
    top_lrs = lrs[n_sig>min_sig_spots][0:n_top]
    top_genes = np.unique([lr.split('_') for lr in top_lrs])

    ## Determining the background genes if not inputted ##
    if type(bg_genes)==type(None):
        all_lrs = load_lrs('connectomeDB2020_put')
        bg_genes = np.unique([lr_.split('_') for lr_ in all_lrs])

    #### Running the GO analysis ####
    go_results = run_GO(top_genes, bg_genes, species, r_path,
                        p_cutoff=p_cutoff, q_cutoff=q_cutoff, onts=onts)
    adata.uns['lr_go'] = go_results
    if verbose:
        print("GO results saved to adata.uns['lr_go']")

################################################################################
            # Functions for calling Celltype-Celltype interactions #
################################################################################
def run_cci(adata: AnnData, use_label: str, spot_mixtures: bool = False,
            min_spots: int = 3, sig_spots: bool = True,
            cell_prop_cutoff: float = 0.2, p_cutoff=.05, n_perms=100,
            verbose: bool = True,
            ):
    """ Calls significant celltype-celltype interactions based on cell-type data randomisation.
    Parameters
    ----------
    adata: AnnData          Must have had st.tl.cci_rank.run() called prior.
    use_label: str          If !spot_mixtures, is a key in adata.obs, else key in adata.obsm.
    spot_mixtures: bool     If true, indicates using deconvolution data, hence use_label refers to adata.obsm.
    min_spots: int          Specifies the minimum number of spots where LR score present to include in subsequent analysis.
    sig_spots: bool         If true, only consider edges which include a signficant spot from calling st.tl.run()
    cell_prop_cutoff: float Only relevant if spot_mixtures==True, indicates cutoff where cell type considered found in spot.
    p_cutoff: float         Value at which p is considered significant.
    n_perms: int            Number of randomisations of cell data to generate p-values.
    verbose: bool           True if print dialogue to user during run-time.
    Returns
    -------
    adata: AnnData          Relevant information stored: adata.uns[f'*_use_label']
    """
    ran_lr = 'lr_summary' in adata.uns
    ran_sig = False if not ran_lr else 'n_spots_sig' \
									   		  in adata.uns['lr_summary'].columns
    if not ran_lr and not ran_sig:
        raise Exception("No LR results testing results found, " 
                        "please run st.tl.cci_rank.run first")

    # Ensuring compatibility with current way of adding label_transfer to object
    if use_label == "label_transfer" or use_label == "predictions":
        obs_key, uns_key = "predictions", "label_transfer"
    else:
        obs_key, uns_key = use_label, use_label

    # Getting the cell/tissue types that we are actually testing #
    if obs_key not in adata.obs:
        raise Exception(f"Missing {obs_key} from adata.obs, need this even if "
                        f"using mixture mode.")
    tissue_types = adata.obs[obs_key].values.astype(str)
    all_set = np.unique(tissue_types)

    # Mixture mode
    mix_mode = spot_mixtures and uns_key in adata.uns
    if not mix_mode and spot_mixtures:
        print(f"Warning: specified spot_mixtures but no deconvolution data in "
              f"adata.uns['{uns_key}'].\nFalling back to discrete mode.")
    if mix_mode: # Checking the deconvolution results stored correctly.
        cols_present = np.all([cell_type in adata.uns[uns_key]
                                                      for cell_type in all_set])
        rows_present = np.all(adata.uns[uns_key
                                         ].index.values==adata.obs_names.values)
        msg = f"Cell type scores misformatted in adata.uns[{uns_key}]:\n"
        if not cols_present or not rows_present:
            if not cols_present:
                msg = msg+f"Cell types missing from adata.uns[{uns_key}] columns:\n" \
             f"{[cell for cell in all_set if cell not in adata.uns[uns_key]]}\n"
            elif not rows_present:
                msg = msg+"Rows do not correspond to adata.obs_names.\n"
            raise Exception(msg)

    # Getting minimum necessary information for edge counting #
    spot_bcs, cell_data, neighbourhood_bcs, neighbourhood_indices = \
                            get_data_for_counting(adata, use_label,
                                                              mix_mode, all_set)

    lr_summary = adata.uns['lr_summary']
    col_i = 1 if sig_spots else 0
    col = 'lr_sig_scores'if sig_spots else 'lr_scores'
    best_lr_indices = np.where( lr_summary.values[:,col_i] > min_spots )[0]
    best_lrs = lr_summary.index.values[ best_lr_indices ]
    lr_genes = np.unique([lr.split('_') for lr in best_lrs])
    lr_expr = adata[:,lr_genes].to_df()

    # Sig-CCIs across all LRs #
    all_matrix = np.zeros((len(all_set), len(all_set)), dtype=int)
    # CCIs across all LRs #
    raw_matrix = np.zeros((len(all_set), len(all_set)), dtype=int)
    per_lr_cci = {} # Per LR significant CCI counts #
    per_lr_cci_pvals = {} # Per LR CCI p-values #
    per_lr_cci_raw = {} # Per LR raw CCI counts #
    lr_n_spot_cci = np.zeros(( lr_summary.shape[0] ))
    lr_n_spot_cci_sig = np.zeros(( lr_summary.shape[0] ))
    lr_n_cci_sig = np.zeros(( lr_summary.shape[0] ))
    for i, best_lr in enumerate(best_lrs):
        l, r = best_lr.split('_')

        L_bool = lr_expr.loc[:,l].values > 0
        R_bool = lr_expr.loc[:,r].values > 0
        lr_index = np.where(adata.uns['lr_summary'].index.values==best_lr)[0][0]
        sig_bool = adata.obsm[col][:, lr_index] > 0

        int_matrix = get_interaction_matrix(cell_data, neighbourhood_bcs,
                                            neighbourhood_indices, all_set,
                                            sig_bool, L_bool, R_bool,
                                                   cell_prop_cutoff).astype(int)

        int_pvals = get_interaction_pvals(int_matrix, n_perms, cell_data,
                          neighbourhood_bcs, neighbourhood_indices, all_set,
                                     sig_bool, L_bool, R_bool, cell_prop_cutoff)

        # Setting spot counts to 0 for non-significant ccis #
        sig_int_matrix = int_matrix.copy()
        sig_int_matrix[ int_pvals>p_cutoff ] = 0

        # Summarising n-spots sig/non-sig across ccis & total cci_rank for lr pair
        lr_n_spot_cci[ best_lr_indices[i] ] = int_matrix.sum()
        lr_n_spot_cci_sig[ best_lr_indices[i] ] = sig_int_matrix.sum()
        lr_n_cci_sig[best_lr_indices[i]] = (int_pvals<p_cutoff).sum()

        raw_matrix += int_matrix
        all_matrix += sig_int_matrix
        int_df = pd.DataFrame(int_matrix, index=all_set, columns=all_set)
        sig_int_df = pd.DataFrame(sig_int_matrix, index=all_set, columns=all_set)
        pval_df = pd.DataFrame(int_pvals, index=all_set, columns=all_set)
        per_lr_cci[best_lr] = sig_int_df
        per_lr_cci_pvals[best_lr] = pval_df
        per_lr_cci_raw[best_lr] = int_df

    # Saving results to anndata #
    adata.uns['lr_summary'][f'n_cci_sig_{use_label}'] = lr_n_cci_sig
    adata.uns['lr_summary'][f'n-spot_cci_{use_label}'] = lr_n_spot_cci
    adata.uns['lr_summary'][f'n-spot_cci_sig_{use_label}'] = lr_n_spot_cci_sig
    adata.uns[f'lr_cci_{use_label}'] = pd.DataFrame(all_matrix,
                                                 index=all_set, columns=all_set)
    adata.uns[f'lr_cci_raw_{use_label}'] = pd.DataFrame(raw_matrix,
                                                 index=all_set, columns=all_set)
    adata.uns[f'per_lr_cci_{use_label}'] = per_lr_cci
    adata.uns[f'per_lr_cci_pvals_{use_label}'] = per_lr_cci_pvals
    adata.uns[f'per_lr_cci_raw_{use_label}'] = per_lr_cci_raw
    if verbose:
        print(f"Significant counts of cci_rank interactions for all LR pairs in "
              f"{f'lr_cci_{use_label}'}")
        print(f"Significant counts of cci_rank interactions for each LR pair "
              f"stored in dictionary {f'per_lr_cci_{use_label}'}")






