"""Wrapper function for performing CCI analysis, varrying the analysis based on
the inputted data / state of the anndata object.
"""

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from anndata import AnnData
from scipy import sparse
from tqdm import tqdm

from ...utils import _set_n_threads
from ._results import (
    check_lr_results,
    correct_pvals,
    lr_result,
    result_matrix,
    store_lr_results,
)
from .go import run_go
from .het import (
    count,
    get_data_for_counting,
    get_interaction_matrix,
    get_interaction_pvals,
    label_keys,
)
from .ligand_receptors import LRPairs, lr_pairs
from .neighbours import calc_distance, calc_neighbours, get_neighbourhoods
from .permutation import perform_spot_testing
from .scoring import get_lrs_scores


def _report_dropped_lrs(
    adata: AnnData,
    lrs: npt.NDArray[np.str_],
    n_scored: npt.NDArray[np.int64],
    min_spots: int,
    min_expr: float,
    max_named: int = 10,
) -> None:
    """Prints why LR pairs are dropped: a gene with no expression above min_expr
    (with min_expr 0, such a pair scores in no spot; above 0 it can still score
    through its partner, so only dropped pairs are looked at), or else scored in
    min_spots spots or fewer."""
    dropped = lrs[n_scored <= min_spots]
    by_gene, unexpressed = [], set()
    if dropped.size:
        genes = np.unique([lr.split("_") for lr in dropped])
        n_above = np.asarray((min_expr < adata[:, genes].X).sum(axis=0)).ravel()
        unexpressed = set(genes[n_above == 0])
        by_gene = [lr for lr in dropped if unexpressed.intersection(lr.split("_"))]
    too_few = dropped.size - len(by_gene)
    if by_gene:
        named = ", ".join(sorted(unexpressed)[:max_named])
        more = (
            f" and {len(unexpressed) - max_named} more"
            if len(unexpressed) > max_named
            else ""
        )
        print(
            f"Dropping {len(by_gene)} LR pairs with a gene never expressed above "
            f"min_expr={min_expr}: {named}{more}."
        )
    if too_few:
        print(f"Dropping {too_few} LR pairs scored in {min_spots} or fewer spots.")
    print(f"Altogether {lrs.size - dropped.size} valid L-R pairs")


def run(
    adata: AnnData,
    lrs: LRPairs,
    min_spots: int = 10,
    distance: float | None = None,
    n_pairs: int = 1000,
    n_cpus: int | None = None,
    use_label: str | None = None,
    adj_method: str = "fdr_bh",
    pval_adj_cutoff: float = 0.05,
    min_expr: float = 0.0,
    save_bg: bool = False,
    neg_binom: bool = False,
    random_state: int = 0,
    verbose: bool = True,
    cell_prop_cutoff: float = 0.2,
) -> AnnData:
    """Performs stLearn LR analysis.

    Parameters
    ----------
    adata: AnnData
        The data object.
    lrs: LRPairs
        The LR pairs to score/test for enrichment, e.g. st.tl.cci.lr_pairs() for
        connectomeDB2025 or lr_pairs(species="mouse") for mouse data. Each pair's
        source goes in the LR table's var['source'].
    min_spots: int
        More than min_spots with an LR score for an LR to be considered for
        further testing.
    distance: float
        Distance to determine the neighbours (default [None] is immediately
        adjacent neighbours if using Visium), distance=0 means within spot
        (only for non-single-cell spatial data).
    n_pairs: int
        Number of random pairs of genes to generate when creating the background
        distribution per LR pair; higher than more accurate p-value estimation.
    n_cpus: int
        Number of threads to use or if None use numba.config.NUMBA_NUM_THREADS
    use_label: str
        The cell type deconvolution results to use in counting stored in
        adata.uns or .obs; if not specified only considered LR expression without cell
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
    random_state:
        A random seed for reproducibility.
    verbose: bool
        Whether print dialogue to user during run-time.
    cell_prop_cutoff: float
        With use_label, a cell type is in a spot when its proportion is above this,
        as in run_cci.
    Returns
    -------
    AnnData
        The LR table: the spots of adata as rows and the LR pairs as columns,
        most significant first. Pass it to the functions that use the results
        as lr_results=, and keep it in a SpatialData store as a table of its own
        beside adata's.
            var
                Summary of significant spots detected per LR: 'n_spots',
                'n_spots_sig' and 'n_spots_sig_pval'; and with LRPairs, 'source',
                where each pair came from.
            layers
                'lr_scores', 'lr_sig_scores', 'p_vals', 'p_adjs' and
                'neg_log10_p_adjs', spots by LR pairs. 'lr_scores' is the raw
                scores, while 'lr_sig_scores' is the same except only for
                significant scores; non-significant scores are set to zero.
                The layers are sparse (CSC), with an entry for each spot
                tested for an LR: each spot with a score above 0 for it. The
                other spots have a score of 0 and a p-value of 1, which the
                matrices leave out; read a layer with st.tl.cci.lr_result,
                which fills them in, one LR at a time for a large sample.
            uns
                Returns 'lr_species' and 'lrfeatures'
                With save_bg 'lrs_to_bg' and 'lr_spot_indices'.
        Relevant information stored in adata:
            adata.obsm['spot_neighbours'] and adata.obsm['spot_neigh_bcs']
                The neighbours of each spot.
            adata.obsm['cci_het']
                Only if use_label specified; contains the counts of the cell types found
                per spot.
    Raises
    ------
    ValueError:
        If gene names contain '_', or LR pairs were not found or did not score
        sufficiently in a minimum number of spots.
    TypeError:
        If lrs aren't LRPairs.
    """

    if not isinstance(lrs, LRPairs):
        raise TypeError(
            f"lrs must be LRPairs, got {type(lrs).__name__}; use lr_pairs(database), "
            "read_lr_pairs(path), LRPairs.from_frame(df) or LRPairs.from_names(names, "
            "species=...)."
        )

    if len(lrs) == 0:
        raise ValueError("No LR pairs given.")

    _set_n_threads(n_cpus)

    # Making sure none of the var_names contains '_' already, these will need
    # to be renamed.
    prob_genes = [gene for gene in adata.var_names if "_" in gene]
    if len(prob_genes) > 0:
        raise ValueError(
            "Detected '_' within some gene names, which breaks "
            + "internal string handling for the lrs in format 'L_R'.\n"
            + "Recommend to rename adata.var_names or remove these "
            + f"genes from adata:\n {prob_genes}",
        )

    # Calculating neighbour & storing #
    distance = calc_distance(adata, distance)
    neighbours = calc_neighbours(adata, distance, verbose=verbose)
    adata.obsm["spot_neighbours"] = pd.DataFrame(
        [",".join(x.astype(str)) for x in neighbours],
        index=adata.obs_names,
        columns=["neighbour_indices"],
    )

    # Important to store barcodes in-case adata subsetted #
    barcodes = adata.obs_names.to_numpy()
    adata.obsm["spot_neigh_bcs"] = pd.DataFrame(
        list(map(lambda x: ",".join(barcodes[x]), neighbours)),
        index=adata.obs_names,
        columns=["neighbour_bcs"],
    )

    if verbose:
        print(
            "Spot neighbour indices stored in adata.obsm['spot_neighbours'] "
            "& adata.obsm['spot_neigh_bcs'].",
        )

    # Weighting by cell heterogeneity if given cell types
    if use_label is not None:
        if verbose:
            print("Calculating cell heterogeneity...")

        # Calculating cell heterogeneity #
        count(
            adata,
            distance=distance,
            use_label=use_label,
            cell_prop_cutoff=cell_prop_cutoff,
            verbose=verbose,
        )
        het_vals = adata.obsm["cci_het"].ravel()
    else:
        het_vals = np.ones(len(adata), dtype=int)

    # 1. Filter any LRs without stored expression.
    # Calculating the lr_scores across spots for the inputted lrs #
    lr_scores, new_lrs = get_lrs_scores(
        adata, lrs, neighbours, het_vals, min_expr, sparse_result=True
    )
    # Fail fast.
    if new_lrs.size == 0:
        raise ValueError(
            f"None of the {len(lrs)} {lrs.species} LR pairs are in adata.var_names; "
            "check the pairs' species matches the data, e.g. "
            "lr_pairs(species='mouse')."
        )
    n_scored = np.diff(lr_scores.indptr).astype(np.int64)  # spots scored per LR
    lr_bool = n_scored > min_spots
    if verbose:
        _report_dropped_lrs(adata, new_lrs, n_scored, min_spots, min_expr)
    new_lrs = new_lrs[lr_bool]
    lr_scores = lr_scores[:, lr_bool]
    if new_lrs.size == 0:
        raise ValueError(
            f"No LR pairs had scores in more than {min_spots} spots; "
            "try lowering min_spots or min_expr."
        )
    # Permutation methods generating background per spot, & test lrs in spot.
    tests = perform_spot_testing(
        adata,
        lr_scores,
        new_lrs,
        n_pairs,
        neighbours,
        het_vals,
        min_expr,
        verbose,
        save_bg=save_bg,
        neg_binom=neg_binom,
        random_state=random_state,
    )
    lr_results = AnnData(
        obs=pd.DataFrame(index=adata.obs_names), var=pd.DataFrame(index=new_lrs)
    )

    lr_results.var["source"] = lrs.source_of(new_lrs).astype(str)
    lr_results.uns["lr_species"] = lrs.species
    lr_results.uns["lrfeatures"] = tests.lr_features

    padjs = correct_pvals(lr_scores, tests.pvals, "spot", adj_method)
    store_lr_results(lr_results, lr_scores, tests.pvals, padjs, pval_adj_cutoff)
    if save_bg:  # after the table is reordered, which copies uns
        lr_results.uns["lrs_to_bg"] = tests.backgrounds
        lr_results.uns["lr_spot_indices"] = tests.bg_spot_indices

    if verbose:
        print(
            "\nLR results returned as a table: the summary in .var, the per-spot "
            "results in .layers. Pass it to the CCI functions as lr_results=."
        )
    return lr_results


def adj_pvals(
    lr_results: AnnData,
    pval_adj_cutoff: float = 0.05,
    correct_axis: str | None = "spot",
    adj_method: str = "fdr_bh",
    verbose: bool = True,
) -> None:
    """Performs p-value adjustment and determination of significant spots.
        Default settings of this function are already run in st.tl.cci.run.

    Parameters
    ----------
    lr_results: AnnData
        The LR table st.tl.cci.run returned.
    pval_adj_cutoff: float
        Cutoff for spot to be significant based on adjusted p-value.
    correct_axis: str
        Either 'spot', 'LR', or None; former corrects for number of LRs tested
        in each spot, middle corrects for number of spots tested per LR, and
        latter performs no adjustment (uses p-value for significant testing)
    adj_method: str
        Any method supported by statsmodels.stats.multitest.multipletests;
        https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
    verbose: bool
        Whether to print what was updated.
    Returns
    -------
    None
        Updates the LR table's summary (var) and results (layers), with LRs
        re-ranked by their new number of significant spots. Columns that
        st.tl.cci.run_cci added to the summary are dropped, since they describe
        the previous results; re-run run_cci to update them.
    """
    lr_scores = result_matrix(lr_results, "lr_scores")
    pvals = result_matrix(lr_results, "p_vals")
    padjs = correct_pvals(lr_scores, pvals, correct_axis, adj_method)
    store_lr_results(lr_results, lr_scores, pvals, padjs, pval_adj_cutoff)
    if verbose:
        print("Updated the LR table's summary (var) and results (layers).")


def run_lr_go(
    lr_results: AnnData,
    r_path: str,
    n_top: int = 100,
    bg_genes: npt.ArrayLike | None = None,
    min_sig_spots: int = 1,
    species: str | None = None,
    p_cutoff: float = 0.01,
    q_cutoff: float = 0.5,
    onts: str = "BP",
    verbose: bool = True,
):
    """Runs a basic GO analysis on the genes in the top ranked LR pairs.
        Only supported for human and mouse species.

    Parameters
    ----------
    lr_results: AnnData
        The LR table st.tl.cci.run returned.
    r_path: str
        Path to R, must have clusterProfiler, org.Mm.eg.db, and org.Hs.eg.db
        installed.
    bg_genes: array-like
        Genes to be used as the background, e.g. lrs.genes() for the LRPairs given
        to run. If None, defaults to all genes in lr_pairs(species=species)
        (connectomeDB2025).
    n_top: int
        The top number of LR pairs to use.
    min_sig_spots: int
        Minimum no. of significant spots pairs must have to be considered.
    species: str
        Species to perform the GO testing for. If None, the species of the LRPairs
        given to run (uns['lr_species']), or 'human'.
    p_cutoff: float
        P-value & P-adj cutoff below which results will be returned.
    q_cutoff: float
        Q-value cutoff below which results will be returned.
    onts: str
        As per clusterProfiler; One of "BP", "MF", and "CC" subontologies, or "ALL"
        for all three.
    Returns
    -------
    None
        The GO terms are stored in the LR table's uns['lr_go'].
    """
    # Making sure inputted correct species
    check_lr_results(lr_results)
    pairs_species = lr_results.uns.get("lr_species")
    if species is None:
        species = pairs_species or "human"
    elif pairs_species is not None and species != pairs_species:
        raise ValueError(
            f"species is {species!r} but the LR table's pairs are {pairs_species}."
        )
    all_species = ["human", "mouse"]
    if species not in all_species:
        raise ValueError(f"Got {species} for species, must be one of {all_species}")

    # Getting the genes from the top LR pairs
    lrs = lr_results.var_names.to_numpy().astype(str)
    n_sig = lr_results.var["n_spots_sig"].to_numpy().astype(int)
    top_lrs = lrs[n_sig > min_sig_spots][0:n_top]
    top_genes = np.unique([lr.split("_") for lr in top_lrs])

    # Determining the background genes if not inputted
    if bg_genes is None:
        bg_genes = lr_pairs(species=species).genes()
    bg_genes = np.asarray(bg_genes, dtype=np.str_)

    # Running the GO analysis
    go_results = run_go(
        top_genes,
        bg_genes,
        species,
        r_path,
        p_cutoff=p_cutoff,
        q_cutoff=q_cutoff,
        onts=onts,
    )
    lr_results.uns["lr_go"] = go_results
    if verbose:
        print("GO results saved to the LR table's uns['lr_go']")


# Functions for calling Celltype-Celltype interactions
def run_cci(
    adata: AnnData,
    use_label: str,
    lr_results: AnnData | None = None,
    spot_mixtures: bool = False,
    min_spots: int = 3,
    sig_spots: bool = True,
    cell_prop_cutoff: float = 0.2,
    p_cutoff: float = 0.05,
    n_perms: int = 100,
    n_cpus: int | None = None,
    random_state: int = 0,
    verbose: bool = True,
):
    """Calls significant celltype-celltype interactions based on cell-type data
    randomisation.

    Parameters
    ----------
    adata: AnnData
        Must have had st.tl.cci.run() called prior.
    use_label: str
        If !spot_mixtures, is a key in adata.obs, else key in adata.uns.
        Note if spot_mixtures specified, must have both the deconvolution data
        in adata.uns[use_label] and the dominant cell type per spot stored in
        adata.obs[use_label]. See tutorial for example.
    lr_results: AnnData
        The LR table st.tl.cci.run returned for adata.
    spot_mixtures: bool
        If true, indicates using deconvolution data, hence use_label
        refers to adata.uns.
    min_spots: int
        Specifies more than min_spots where LR score present to include in
        subsequent analysis.
    sig_spots: bool
        If true, only consider edges which include a signficant spot from
        calling st.tl.cci.run()
    cell_prop_cutoff: float
        Only relevant if spot_mixtures==True, indicates cutoff where cell type
        considered found in spot.
    p_cutoff: float
        Value at which p is considered significant.
    n_perms: int
        Number of randomisations of cell data to generate p-values. A p-value is
        (permutations with at least the observed count + 1) / (n_perms + 1), so
        the smallest is 1 / (n_perms + 1). If set to 0, then performs no
        permutations, but still does perform raw counting of the cell type
        interactions with each LR hotspot. This can still be visualised
        downstream by setting paramters to plot significant interactions to false.
    n_cpus: int | None
        cpu resources to use.
    random_state:
        A random seed for reproducibility.
    verbose: bool
        True if print dialogue to user during run-time.
    Returns
    -------
    None
        Relevant information stored in the LR table
            var
                Additional columns; f"n_cci_sig_{use_label}",
                f"n-spot_cci_{use_label}", f"n-spot_cci_sig_{use_label}".
                Former is the no. of CCIs significant for the LR, middle is
                the no. of individual spot-spot interactions across all CCIs for
                LR, and latter is the no. of significant individual spot
                interactions.
            uns
                Dataframes added:
                    f"lr_cci_{use_label}"
                        The count of significant spot-spot interactions across
                        all LR pairs for each possible CCI.
                    f"lr_cci_raw_{use_label}"
                        The raw count of spot-spot interactions across all LR
                        pairs for each possible CCI.
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
    _set_n_threads(n_cpus)

    lr = check_lr_results(lr_results, adata)
    if "n_spots_sig" not in lr.var.columns:
        raise ValueError(
            "No LR results testing results found, please run st.tl.cci.run first",
        )

    obs_key, uns_key = label_keys(use_label)

    # Getting the cell/tissue types that we are actually testing #
    if obs_key not in adata.obs:
        raise KeyError(
            f"Missing {obs_key} from adata.obs, need this even if using mixture mode.",
        )
    tissue_types = adata.obs[obs_key].to_numpy().astype(str)
    all_set = np.unique(tissue_types)

    # Mixture mode
    mix_mode = spot_mixtures and uns_key in adata.uns
    if not mix_mode and spot_mixtures:
        warnings.warn(
            f"Specified spot_mixtures but no deconvolution data in "
            f"adata.uns['{uns_key}']. Falling back to discrete mode.",
            stacklevel=2,
        )
    if mix_mode:  # Checking the deconvolution results stored correctly.
        cols_present = np.all(
            [cell_type in adata.uns[uns_key] for cell_type in all_set],
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
            raise ValueError(msg)

        # Checking for case where have cell types that are never dominant
        # in a spot, so need to include these in all_set
        if len(all_set) < adata.uns[uns_key].shape[1]:
            all_set = adata.uns[uns_key].columns.to_numpy().astype(str)

    # Getting minimum necessary information for edge counting
    if verbose:
        print("Getting cached neighbourhood information...")
    # Getting the neighbourhoods #
    _, neighbourhood_indices = get_neighbourhoods(adata)

    if verbose:
        print("Getting information for CCI counting...")

    cell_data = get_data_for_counting(adata, use_label, mix_mode, all_set)

    lr_summary = lr.var
    count_col = "n_spots_sig" if sig_spots else "n_spots"
    col = "lr_sig_scores" if sig_spots else "lr_scores"
    best_lr_indices = np.flatnonzero(lr_summary[count_col].to_numpy() > min_spots)
    best_lrs = lr_summary.index.values[best_lr_indices]
    lr_genes = np.unique([lr.split("_") for lr in best_lrs])
    if len(lr_genes) == 0:
        raise ValueError(
            "No LR pairs returned with current filtering params; \n"
            "may need to adjust min_spots, sig_spots parameters, "
            "or re-run st.tl.cci.run with more relaxed parameters.",
        )
    lr_expr = sparse.csc_matrix(adata[:, lr_genes].X)  # a gene at a time below
    gene_column = {gene: i for i, gene in enumerate(lr_genes)}

    def expressed(gene: str) -> npt.NDArray[np.bool_]:
        return lr_expr[:, gene_column[gene]].toarray().ravel() > 0

    # Sig-CCIs across all LRs #
    all_matrix = np.zeros((len(all_set), len(all_set)), dtype=int)
    # CCIs across all LRs #
    raw_matrix = np.zeros((len(all_set), len(all_set)), dtype=int)
    per_lr_cci = {}  # Per LR significant CCI counts #
    per_lr_cci_pvals = {}  # Per LR CCI p-values #
    per_lr_cci_raw = {}  # Per LR raw CCI counts #
    lr_n_spot_cci = np.zeros(lr_summary.shape[0])
    lr_n_spot_cci_sig = np.zeros(lr_summary.shape[0])
    lr_n_cci_sig = np.zeros(lr_summary.shape[0])
    with tqdm(
        total=len(best_lrs),
        desc="Counting celltype-celltype interactions per LR and permuting "
        + f"{n_perms} times.",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
        disable=not verbose,
    ) as pbar:
        for i, best_lr in enumerate(best_lrs):
            ligand, receptor = best_lr.split("_")

            L_bool = expressed(ligand)
            R_bool = expressed(receptor)
            sig_bool = lr_result(lr, col, int(best_lr_indices[i])) > 0

            int_matrix = get_interaction_matrix(
                cell_data,
                neighbourhood_indices,
                all_set,
                sig_bool,
                L_bool,
                R_bool,
                cell_prop_cutoff,
            ).astype(int)

            if n_perms > 0:
                int_pvals = get_interaction_pvals(
                    int_matrix,
                    n_perms,
                    cell_data,
                    neighbourhood_indices,
                    all_set,
                    sig_bool,
                    L_bool,
                    R_bool,
                    cell_prop_cutoff,
                    random_state + i,
                )
            else:
                int_pvals = np.ones(int_matrix.shape)

            # Setting spot counts to 0 for non-significant ccis #
            sig_int_matrix = int_matrix.copy()
            sig_int_matrix[int_pvals >= p_cutoff] = 0

            # Summarising n-spots sig/non-sig across CCIs for the LR pair
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

            pbar.update(1)

    # Saving results to the LR table #
    lr.var[f"n_cci_sig_{use_label}"] = lr_n_cci_sig
    lr.var[f"n-spot_cci_{use_label}"] = lr_n_spot_cci
    lr.var[f"n-spot_cci_sig_{use_label}"] = lr_n_spot_cci_sig
    lr.uns[f"lr_cci_{use_label}"] = pd.DataFrame(
        all_matrix,
        index=all_set,
        columns=all_set,
    )
    lr.uns[f"lr_cci_raw_{use_label}"] = pd.DataFrame(
        raw_matrix,
        index=all_set,
        columns=all_set,
    )
    lr.uns[f"per_lr_cci_{use_label}"] = per_lr_cci
    lr.uns[f"per_lr_cci_pvals_{use_label}"] = per_lr_cci_pvals
    lr.uns[f"per_lr_cci_raw_{use_label}"] = per_lr_cci_raw
    if verbose:
        print(
            "Significant cell type interaction counts for all LR pairs stored in "
            f"the LR table's uns['lr_cci_{use_label}']",
        )
        print(
            "Significant cell type interaction counts for each LR pair stored in "
            f"the LR table's uns['per_lr_cci_{use_label}']",
        )
