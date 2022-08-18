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
from .het import (
    count,
    get_data_for_counting,
    get_interaction_matrix,
    get_interaction_pvals,
)
from statsmodels.stats.multitest import multipletests

################################################################################
# Functions related to Ligand-Receptor interactions #
################################################################################
def load_lrs(names: Union[str, list, None] = None, species: str = "human") -> np.array:
    """Loads inputted LR database, & concatenates into consistent database set of pairs without duplicates. If None loads 'connectomeDB2020_lit'.

    Parameters
    ----------
    names: list     Databases to load, options: 'connectomeDB2020_lit' (literature verified), 'connectomeDB2020_put' (putative). If more than one specified, loads all & removes duplicates.
    species: str    Format of the LR genes, either 'human' or 'mouse'.
    Returns
    -------
    lrs: np.array   lr pairs from the database in format ['L1_R1', 'LN_RN']
    """
    if type(names) == type(None):
        names = ["connectomeDB2020_lit"]
    if type(names) == str:
        names = [names]

    path = os.path.dirname(os.path.realpath(__file__))
    dbs = [pd.read_csv(f"{path}/databases/{name}.txt", sep="\t") for name in names]
    lrs_full = []
    for db in dbs:
        lrs = [f"{db.values[i,0]}_{db.values[i,1]}" for i in range(db.shape[0])]
        lrs_full.extend(lrs)
    lrs_full = np.unique(lrs_full)
    # If dealing with mouse, need to reformat #
    if species == "mouse":
        genes1 = [lr_.split("_")[0] for lr_ in lrs_full]
        genes2 = [lr_.split("_")[1] for lr_ in lrs_full]
        lrs_full = np.array(
            [
                genes1[i][0]
                + genes1[i][1:].lower()
                + "_"
                + genes2[i][0]
                + genes2[i][1:].lower()
                for i in range(len(lrs_full))
            ]
        )

    return lrs_full


def grid(adata, n_row: int = 10, n_col: int = 10, use_label: str = None):
    """Creates a new anndata representing a gridded version of the data; can be
        used upstream of CCI pipeline. NOTE: intended use is for single cell
        spatial data, not Visium or other lower resolution tech.

    Parameters
    ----------
    adata: AnnData
        The data object.
    n_row: int
        The number of rows in the grid.
    n_col: int
        The number of columns in the grid.
    use_label: str
        The cell type labels in adata.obs to join together & save as deconvolution data.
    Returns
    -------
    grid_data: AnnData
        Equivalent expression data to adata, except values have been summed by
        cells that fall within defined bins.
    """

    # Retrieving the coordinates of each grid #
    n_squares = n_row * n_col
    cell_bcs = adata.obs_names.values
    xs, ys = adata.obs["imagecol"].values, adata.obs["imagerow"].values

    grid_counts, xedges, yedges = np.histogram2d(xs, ys, bins=[n_col, n_row])

    grid_expr = np.zeros((n_squares, adata.shape[1]))
    grid_coords = np.zeros((n_squares, 2))
    grid_bcs = []
    grid_cell_counts = []
    gridded_cells = []
    cell_grid = []
    # If use_label specified, then will generate deconvolution information
    if type(use_label) != type(None):
        cell_labels = adata.obs[use_label].values.astype(str)
        cell_set = np.unique(cell_labels)
        cell_info = np.zeros((n_squares, len(cell_set)))

    # generate grids from top to bottom and left to right
    n = 0
    for i in range(n_col):
        x_left, x_right = xedges[i], xedges[i + 1]
        for j in range(n_row):
            y_down, y_up = yedges[j], yedges[j + 1]
            grid_coords[n, :] = [(x_right + x_left) / 2, (y_up + y_down) / 2]

            # Now determining the cells within the gridded area #
            if i != n_col - 1 and j == n_row - 1:  # top left corner
                x_true = (xs >= x_left) & (xs < x_right)
                y_true = (ys <= y_up) & (ys > y_down)
            elif i == n_col - 1 and j != n_row - 1:  # bottom righ corner
                x_true = (xs > x_left) & (xs <= x_right)
                y_true = (ys < y_up) & (ys >= y_down)
            else:  # average case
                x_true = (xs >= x_left) & (xs < x_right)
                y_true = (ys < y_up) & (ys >= y_down)

            grid_cells = cell_bcs[x_true & y_true]
            grid_cells_str = ",".join(grid_cells)
            grid_bcs.append(grid_cells_str)
            grid_cell_counts.append(len(grid_cells))
            gridded_cells.extend(grid_cells)
            cell_grid.extend([f"grid_{n}"] * len(grid_cells))

            # Summing the expression across these cells to get the grid expression #
            if len(grid_cells) > 0:
                cell_bool = [cell in grid_cells for cell in cell_bcs]
                grid_expr[n, :] = adata.X[cell_bool, :].sum(axis=0)

            # If we have cell type information, will record #
            if type(use_label) != type(None) and len(grid_cells) > 0:
                grid_cell_types = cell_labels[cell_bool]
                cell_info[n, :] = [
                    len(np.where(grid_cell_types == ct)[0]) / len(grid_cell_types)
                    for ct in cell_set
                ]

            n += 1

    # Creating gridded anndata #
    grid_expr = pd.DataFrame(
        grid_expr,
        index=[f"grid_{i}" for i in range(n_squares)],
        columns=adata.var_names.values.astype(str),
    )
    grid_data = AnnData(grid_expr)
    grid_data.obs["imagecol"] = grid_coords[:, 0]
    grid_data.obs["imagerow"] = grid_coords[:, 1]
    grid_data.obs["n_cells"] = grid_cell_counts
    grid_data.obsm["spatial"] = grid_coords
    grid_data.uns["spatial"] = adata.uns["spatial"]

    if type(use_label) != type(None):
        grid_data.uns[use_label] = pd.DataFrame(
            cell_info, index=grid_data.obs_names.values.astype(str), columns=cell_set
        )
        max_indices = np.apply_along_axis(np.argmax, 1, cell_info)
        cell_set = np.unique(grid_data.uns[use_label].columns.values)
        grid_data.obs[use_label] = [cell_set[index] for index in max_indices]
        grid_data.obs[use_label] = grid_data.obs[use_label].astype("category")

    # Subsetting to only gridded spots that contain cells #
    grid_data = grid_data[grid_data.obs["n_cells"] > 0, :].copy()
    if type(use_label) != type(None):
        grid_data.uns[use_label] = grid_data.uns[use_label].loc[grid_data.obs_names, :]

    grid_data.uns["grid_counts"] = grid_counts
    grid_data.uns["grid_xedges"] = xedges
    grid_data.uns["grid_yedges"] = yedges

    return grid_data


def run(
    adata: AnnData,
    lrs: np.array,
    min_spots: int = 10,
    distance: int = None,
    n_pairs: int = 1000,
    n_cpus: int = None,
    use_label: str = None,
    adj_method: str = "fdr_bh",
    pval_adj_cutoff: float = 0.05,
    min_expr: float = 0,
    save_bg: bool = False,
    neg_binom: bool = False,
    verbose: bool = True,
):
    """Performs stLearn LR analysis.

    Parameters
    -----------
    adata: AnnData
        The data object.
    lrs: np.array
        The LR pairs to score/test for enrichment (in format 'L1_R1').
    min_spots: int
        Minimum number of spots with an LR score for an LR to be considered for
        further testing.
    distance: int
        Distance to determine the neighbours (default [None] is immediately
        adjacent neighbours if using Visium), distance=0 means within spot
        (only for non-single-cell spatial data).
    n_pairs: int
        Number of random pairs of genes to generate when creating the background
        distribution per LR pair; higher than more accurate p-value estimation.
    n_cpus: int
        The number of cpus to use for multi-threading.
    use_label: str
        The cell type deconvolution results to use in counting stored in
        adata.uns; if not specified only considered LR expression without cell
        heterogeneity.
    adj_method: str
        Parsed to statsmodels.stats.multitest.multipletests for multiple
        hypothesis testing correction; see there for other options.
    pval_adj_cutoff: float
        P-value below which LR is considered significant in spot neighbourhood.
    min_expr: float
        Minimum gene expression of either L or R for spot to be considered to
        expression of either.
    save_bg: bool
        Whether to save the background per LR pair; for method development only.
        Not recommended since huge memory.
    neg_binom: bool
        Whether to fit a negative binomial distribution for all background
        scores generated across spots per LR after discretising the random
        scores. Can be extremely slow.
    verbose: bool
        Whether print dialogue to user during run-time.
    Returns
    --------
    adata: AnnData
    Relevant information stored:
        adata.uns['lr_summary']
            Summary of significant spots detected per LR,
            the LRs listed in the index is the same order of LRs in the columns of
            results stored in adata.obsm below. Hence the order of this must be maintained.
        adata.obsm
            Additional keys are added; 'lr_scores', 'lr_sig_scores', 'p_vals',
            'p_adjs', '-log10(p_adjs)'. All are numpy matrices, with columns
            referring to the LRs listed in adata.uns['lr_summary']. 'lr_scores'
            is the raw scores, while 'lr_sig_scores' is the same except only for
            significant scores; non-significant scores are set to zero.
        adata.obsm['het']
            Only if use_label specified; contains the counts of the cell types found per spot.
    """
    # Setting threads for paralellisation #
    if type(n_cpus) != type(None):
        numba.set_num_threads(n_cpus)

    # Making sure none of the var_names contains '_' already, these will need
    # to be renamed.
    prob_genes = [gene for gene in adata.var_names if "_" in gene]
    if len(prob_genes) > 0:
        raise Exception(
            "Detected '_' within some gene names, which breaks "
            + "internal string handling for the lrs in format 'L_R'.\n"
            + "Recommend to rename adata.var_names or remove these "
            + f"genes from adata:\n {prob_genes}"
        )

    # Calculating neighbour & storing #
    distance = calc_distance(adata, distance)
    neighbours = calc_neighbours(adata, distance, verbose=verbose)
    adata.obsm["spot_neighbours"] = pd.DataFrame(
        [",".join(x.astype(str)) for x in neighbours],
        index=adata.obs_names,
        columns=["neighbour_indices"],
    )
    spot_neighs_df = adata.obsm["spot_neighbours"]
    spot_neigh_bcs = []
    for i in range(spot_neighs_df.shape[0]):
        neigh_indices = [
            int(index)
            for index in spot_neighs_df.values[i, 0].split(",")
            if index != ""
        ]
        neigh_bcs = [adata.obs_names[index] for index in neigh_indices]
        spot_neigh_bcs.append(",".join(neigh_bcs))
    spot_neigh_bcs_df = pd.DataFrame(
        spot_neigh_bcs, index=spot_neighs_df.index, columns=["neighbour_bcs"]
    )
    # Important to store barcodes in-case adata subsetted #
    adata.obsm["spot_neigh_bcs"] = spot_neigh_bcs_df

    if verbose:
        print(
            "Spot neighbour indices stored in adata.obsm['spot_neighbours'] "
            "& adata.obsm['spot_neigh_bcs']."
        )

    # Conduct with cell heterogeneity info if label_transfer provided #
    cell_het = type(use_label) != type(None) and use_label in adata.uns.keys()
    if cell_het:
        if verbose:
            print("Calculating cell hetereogeneity...")

        # Calculating cell heterogeneity #
        count(adata, distance=distance, use_label=use_label, use_het=use_label)

    het_vals = (
        np.array([1] * len(adata))
        if use_label not in adata.obsm
        else adata.obsm[use_label]
    )

    """ 1. Filter any LRs without stored expression.
    """
    # Calculating the lr_scores across spots for the inputted lrs #
    lr_scores, lrs = get_lrs_scores(adata, lrs, neighbours, het_vals, min_expr)
    lr_bool = (lr_scores > 0).sum(axis=0) > min_spots
    lrs = lrs[lr_bool]
    lr_scores = lr_scores[:, lr_bool]
    if verbose:
        print("Altogether " + str(len(lrs)) + " valid L-R pairs")
    if len(lrs) == 0:
        print("Exiting due to lack of valid LR pairs.")
        return

    """ Permutation methods generating background per spot, & test lrs in spot.
    """
    perform_spot_testing(
        adata,
        lr_scores,
        lrs,
        n_pairs,
        neighbours,
        het_vals,
        min_expr,
        adj_method,
        pval_adj_cutoff,
        verbose,
        save_bg=save_bg,
        neg_binom=neg_binom,
    )


def adj_pvals(
    adata,
    pval_adj_cutoff: float = 0.05,
    correct_axis: str = "spot",
    adj_method: str = "fdr_bh",
):
    """Performs p-value adjustment and determination of significant spots.
        Default settings of this function are already run in st.tl.cci.run.

    Parameters
    ----------
    adata: AnnData
        Must have st.tl.cci.run performed prior.
    pval_adj_cutoff: float
        Cutoff for spot to be significant based on adjusted p-value.
    correct_axis: str
        Either 'spot', 'LR', or None; former corrects for number of LRs tested
        in each spot, middle corrects for number of spots tested per LR, and
        latter performs no adjustment (uses p-value for significant testing)
    adj_method: str
        Any method supported by statsmodels.stats.multitest.multipletests;
        https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
    Returns
    -------
    adata: AnnData
        Adjusts all of the LR results; warning, does not adjust
        celltype-celltype results from running ran st.tl.run_cci downstream.
    """
    if "lr_summary" not in adata.uns:
        raise Exception("Need to run st.tl.cci.run first.")

    scores = adata.obsm["lr_scores"]
    sig_scores = scores.copy()
    ps = adata.obsm["p_vals"]
    padjs = np.ones(ps.shape)
    if correct_axis == "spot":
        for spot_i in range(ps.shape[0]):
            lr_indices = np.where(scores[spot_i, :] > 0)[0]
            if len(lr_indices) > 0:
                spot_ps = ps[spot_i, lr_indices]
                spot_padjs = multipletests(spot_ps, method=adj_method)[1]
                padjs[spot_i, lr_indices] = spot_padjs
                sig_scores[spot_i, lr_indices[spot_padjs >= pval_adj_cutoff]] = 0
    elif correct_axis == "LR":
        for lr_i in range(ps.shape[1]):
            spot_indices = np.where(scores[:, lr_i] > 0)[0]
            if len(spot_indices) > 0:
                lr_ps = ps[spot_indices, lr_i]
                spot_padjs = multipletests(lr_ps, method=adj_method)[1]
                padjs[spot_indices, lr_i] = spot_padjs
                sig_scores[spot_indices[spot_padjs >= pval_adj_cutoff], lr_i] = 0
    elif type(correct_axis) == type(None):
        padjs = ps.copy()
        sig_scores[padjs >= pval_adj_cutoff] = 0
    else:
        raise Exception(
            f"Invalid correct_axis input, must be one of: " f"'LR', 'spot', or None"
        )

    # Counting spots significant per lr #
    lr_counts = (padjs < pval_adj_cutoff).sum(axis=0)
    lr_counts_pval = (ps < pval_adj_cutoff).sum(axis=0)

    # Re-ranking LRs based on these counts & updating LR ordering #
    adata.uns["lr_summary"].loc[:, "n_spots_sig"] = lr_counts
    adata.uns["lr_summary"].loc[:, "n_spots_sig_pval"] = lr_counts_pval
    new_order = np.argsort(-adata.uns["lr_summary"].loc[:, "n_spots_sig"].values)
    adata.uns["lr_summary"] = adata.uns["lr_summary"].iloc[new_order, :]
    print(f"Updated adata.uns[lr_summary]")
    scores_ordered = scores[:, new_order]
    sig_scores_ordered = sig_scores[:, new_order]
    ps_ordered = ps[:, new_order]
    padjs_ordered = padjs[:, new_order]
    log10padjs = -np.log10(padjs_ordered)

    keys = ["lr_scores", "lr_sig_scores", "p_vals", "p_adjs", "-log10(p_adjs)"]
    values = [scores_ordered, sig_scores_ordered, ps_ordered, padjs_ordered, log10padjs]
    for i in range(len(keys)):
        adata.obsm[keys[i]] = values[i]
        print(f"Updated adata.obsm[{keys[i]}]")


def run_lr_go(
    adata: AnnData,
    r_path: str,
    n_top: int = 100,
    bg_genes: np.array = None,
    min_sig_spots: int = 1,
    species: str = "human",
    p_cutoff: float = 0.01,
    q_cutoff: float = 0.5,
    onts: str = "BP",
    verbose: bool = True,
):
    """Runs a basic GO analysis on the genes in the top ranked LR pairs.
        Only supported for human and mouse species.

    Parameters
    ----------
    adata: AnnData
        Must have had st.tl.cci_rank.run() called prior.
    r_path: str
        Path to R, must have clusterProfiler, org.Mm.eg.db, and org.Hs.eg.db
        installed.
    bg_genes: np.array
        Genes to be used as the background. If None, defaults to all genes in
        lr database: 'connectomeDB2020_put'.
    n_top: int
        The top number of LR pairs to use.
    min_sig_spots: int
        Minimum no. of significant spots pairs must have to be considered.
    species: str
        Species to perform the GO testing for.
    p_cutoff: float
        P-value & P-adj cutoff below which results will be returned.
    q_cutoff: float
        Q-value cutoff below which results will be returned.
    onts: str
        As per clusterProfiler; One of "BP", "MF", and "CC" subontologies, or "ALL" for all three.
    Returns
    -------
    adata: AnnData
        Relevant information stored in adata.uns['lr_go']
    """
    #### Making sure inputted correct species ####
    all_species = ["human", "mouse"]
    if species not in all_species:
        raise Exception(f"Got {species} for species, must be one of " f"{all_species}")

    #### Getting the genes from the top LR pairs ####
    if "lr_summary" not in adata.uns:
        raise Exception("Need to run st.tl.cci.run first.")
    lrs = adata.uns["lr_summary"].index.values.astype(str)
    n_sig = adata.uns["lr_summary"].loc[:, "n_spots_sig"].values.astype(int)
    top_lrs = lrs[n_sig > min_sig_spots][0:n_top]
    top_genes = np.unique([lr.split("_") for lr in top_lrs])

    ## Determining the background genes if not inputted ##
    if type(bg_genes) == type(None):
        all_lrs = load_lrs("connectomeDB2020_put")
        bg_genes = np.unique([lr_.split("_") for lr_ in all_lrs])

    #### Running the GO analysis ####
    go_results = run_GO(
        top_genes,
        bg_genes,
        species,
        r_path,
        p_cutoff=p_cutoff,
        q_cutoff=q_cutoff,
        onts=onts,
    )
    adata.uns["lr_go"] = go_results
    if verbose:
        print("GO results saved to adata.uns['lr_go']")


################################################################################
# Functions for calling Celltype-Celltype interactions #
################################################################################
def run_cci(
    adata: AnnData,
    use_label: str,
    spot_mixtures: bool = False,
    min_spots: int = 3,
    sig_spots: bool = True,
    cell_prop_cutoff: float = 0.2,
    p_cutoff=0.05,
    n_perms=100,
    verbose: bool = True,
):
    """Calls significant celltype-celltype interactions based on cell-type data randomisation.

    Parameters
    ----------
    adata: AnnData
        Must have had st.tl.cci_rank.run() called prior.
    use_label: str
        If !spot_mixtures, is a key in adata.obs, else key in adata.uns.
        Note if spot_mixtures specified, must have both the deconvolution data
        in adata.uns[use_label] and the dominant cell type per spot stored in
        adata.obs[use_label]. See tutorial for example.
    spot_mixtures: bool
        If true, indicates using deconvolution data, hence use_label
        refers to adata.uns.
    min_spots: int
        Specifies the minimum number of spots where LR score present to
        include in subsequent analysis.
    sig_spots: bool
        If true, only consider edges which include a signficant spot from
        calling st.tl.cci.run()
    cell_prop_cutoff: float
        Only relevant if spot_mixtures==True, indicates cutoff where cell type
        considered found in spot.
    p_cutoff: float
        Value at which p is considered significant.
    n_perms: int
        Number of randomisations of cell data to generate p-values.
    verbose: bool
        True if print dialogue to user during run-time.
    Returns
    -------
    adata: AnnData
        Relevant information stored
            adata.uns['lr_summary']
                Additional columns; f"n_cci_sig_{use_label}",
                f"n-spot_cci_{use_label}", f"n-spot_cci_sig_{use_label}".
                Former is the no. of CCIs significant for the LR, middle is
                the no. of individual spot-spot interactions across all CCIs for
                LR, and latter is the no. of significant individual spot
                interactions.
            adata.uns
                Dataframes added:
                    f"lr_cci_raw_{use_label}"
                        The raw count of spot-spot interactions across all LR
                        pairs for each possible CCI.
                    f"lr_cci_raw_{use_label}"
                        The count of significant spot-spot interactions across
                        all LR pairs for each possible CCI.
                Dictionaries added:
                    f"per_lr_cci_pvals_{use_label}"
                        Each key refers to a LR, with the value being a dataframe
                        listing the p-values for each potential CCI.
                    f"per_lr_cci_raw_{use_label}"
                        Each key refers to a LR, with the value being a dataframe
                        listing the count of spot-spot interactions via the LR in
                        significant LR neighbourhoods stratified by each
                        celltype-celltype combination.
                    f"per_lr_cci_{use_label}"
                        The same as f"per_lr_cci_raw_{use_label}", except
                        subsetted to significant CCIs.
    """
    ran_lr = "lr_summary" in adata.uns
    ran_sig = False if not ran_lr else "n_spots_sig" in adata.uns["lr_summary"].columns
    if not ran_lr and not ran_sig:
        raise Exception(
            "No LR results testing results found, " "please run st.tl.cci.run first"
        )

    # Ensuring compatibility with current way of adding label_transfer to object
    if use_label == "label_transfer" or use_label == "predictions":
        obs_key, uns_key = "predictions", "label_transfer"
    else:
        obs_key, uns_key = use_label, use_label

    # Getting the cell/tissue types that we are actually testing #
    if obs_key not in adata.obs:
        raise Exception(
            f"Missing {obs_key} from adata.obs, need this even if "
            f"using mixture mode."
        )
    tissue_types = adata.obs[obs_key].values.astype(str)
    all_set = np.unique(tissue_types)

    # Mixture mode
    mix_mode = spot_mixtures and uns_key in adata.uns
    if not mix_mode and spot_mixtures:
        print(
            f"Warning: specified spot_mixtures but no deconvolution data in "
            f"adata.uns['{uns_key}'].\nFalling back to discrete mode."
        )
    if mix_mode:  # Checking the deconvolution results stored correctly.
        cols_present = np.all(
            [cell_type in adata.uns[uns_key] for cell_type in all_set]
        )
        rows_present = np.all(adata.uns[uns_key].index.values == adata.obs_names.values)
        msg = f"Cell type scores misformatted in adata.uns[{uns_key}]:\n"
        if not cols_present or not rows_present:
            if not cols_present:
                msg = (
                    msg + f"Cell types missing from adata.uns[{uns_key}] columns:\n"
                    f"{[cell for cell in all_set if cell not in adata.uns[uns_key]]}\n"
                )
            elif not rows_present:
                msg = msg + "Rows do not correspond to adata.obs_names.\n"
            raise Exception(msg)

        #### Checking for case where have cell types that are never dominant
        #### in a spot, so need to include these in all_set
        if len(all_set) < adata.uns[uns_key].shape[1]:
            all_set = adata.uns[uns_key].columns.values.astype(str)

    # Getting minimum necessary information for edge counting #
    (
        spot_bcs,
        cell_data,
        neighbourhood_bcs,
        neighbourhood_indices,
    ) = get_data_for_counting(adata, use_label, mix_mode, all_set)

    lr_summary = adata.uns["lr_summary"]
    col_i = 1 if sig_spots else 0
    col = "lr_sig_scores" if sig_spots else "lr_scores"
    best_lr_indices = np.where(lr_summary.values[:, col_i] > min_spots)[0]
    best_lrs = lr_summary.index.values[best_lr_indices]
    lr_genes = np.unique([lr.split("_") for lr in best_lrs])
    if len(lr_genes) == 0:
        raise Exception(
            "No LR pairs returned with current filtering params; \n"
            "may need to adjust min_spots, sig_spots parameters, "
            "or re-run st.tl.cci.run with more relaxed parameters."
        )
    lr_expr = adata[:, lr_genes].to_df()

    # Sig-CCIs across all LRs #
    all_matrix = np.zeros((len(all_set), len(all_set)), dtype=int)
    # CCIs across all LRs #
    raw_matrix = np.zeros((len(all_set), len(all_set)), dtype=int)
    per_lr_cci = {}  # Per LR significant CCI counts #
    per_lr_cci_pvals = {}  # Per LR CCI p-values #
    per_lr_cci_raw = {}  # Per LR raw CCI counts #
    lr_n_spot_cci = np.zeros((lr_summary.shape[0]))
    lr_n_spot_cci_sig = np.zeros((lr_summary.shape[0]))
    lr_n_cci_sig = np.zeros((lr_summary.shape[0]))
    for i, best_lr in enumerate(best_lrs):
        l, r = best_lr.split("_")

        L_bool = lr_expr.loc[:, l].values > 0
        R_bool = lr_expr.loc[:, r].values > 0
        lr_index = np.where(adata.uns["lr_summary"].index.values == best_lr)[0][0]
        sig_bool = adata.obsm[col][:, lr_index] > 0

        int_matrix = get_interaction_matrix(
            cell_data,
            neighbourhood_bcs,
            neighbourhood_indices,
            all_set,
            sig_bool,
            L_bool,
            R_bool,
            cell_prop_cutoff,
        ).astype(int)

        int_pvals = get_interaction_pvals(
            int_matrix,
            n_perms,
            cell_data,
            neighbourhood_bcs,
            neighbourhood_indices,
            all_set,
            sig_bool,
            L_bool,
            R_bool,
            cell_prop_cutoff,
        )

        # Setting spot counts to 0 for non-significant ccis #
        sig_int_matrix = int_matrix.copy()
        sig_int_matrix[int_pvals > p_cutoff] = 0

        # Summarising n-spots sig/non-sig across ccis & total cci_rank for lr pair
        lr_n_spot_cci[best_lr_indices[i]] = int_matrix.sum()
        lr_n_spot_cci_sig[best_lr_indices[i]] = sig_int_matrix.sum()
        lr_n_cci_sig[best_lr_indices[i]] = (int_pvals < p_cutoff).sum()

        raw_matrix += int_matrix
        all_matrix += sig_int_matrix
        int_df = pd.DataFrame(int_matrix, index=all_set, columns=all_set)
        sig_int_df = pd.DataFrame(sig_int_matrix, index=all_set, columns=all_set)
        pval_df = pd.DataFrame(int_pvals, index=all_set, columns=all_set)
        per_lr_cci[best_lr] = sig_int_df
        per_lr_cci_pvals[best_lr] = pval_df
        per_lr_cci_raw[best_lr] = int_df

    # Saving results to anndata #
    adata.uns["lr_summary"][f"n_cci_sig_{use_label}"] = lr_n_cci_sig
    adata.uns["lr_summary"][f"n-spot_cci_{use_label}"] = lr_n_spot_cci
    adata.uns["lr_summary"][f"n-spot_cci_sig_{use_label}"] = lr_n_spot_cci_sig
    adata.uns[f"lr_cci_{use_label}"] = pd.DataFrame(
        all_matrix, index=all_set, columns=all_set
    )
    adata.uns[f"lr_cci_raw_{use_label}"] = pd.DataFrame(
        raw_matrix, index=all_set, columns=all_set
    )
    adata.uns[f"per_lr_cci_{use_label}"] = per_lr_cci
    adata.uns[f"per_lr_cci_pvals_{use_label}"] = per_lr_cci_pvals
    adata.uns[f"per_lr_cci_raw_{use_label}"] = per_lr_cci_raw
    if verbose:
        print(
            f"Significant counts of cci_rank interactions for all LR pairs in "
            f"{f'lr_cci_{use_label}'}"
        )
        print(
            f"Significant counts of cci_rank interactions for each LR pair "
            f"stored in dictionary {f'per_lr_cci_{use_label}'}"
        )
