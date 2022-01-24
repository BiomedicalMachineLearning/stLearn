import numpy as np
import pandas as pd
from anndata import AnnData
import scipy.spatial as spatial
from numba.typed import List
from numba import njit, jit

from stlearn.tools.microenv.cci.het_helpers import (
    edge_core,
    get_between_spot_edge_array,
    get_data_for_counting,
    add_unique_edges,
    get_neighbourhoods,
    init_edge_list,
)


def count(
    adata: AnnData,
    use_label: str = None,
    use_het: str = "cci_het",
    verbose: bool = True,
    distance: float = None,
) -> AnnData:
    """Count the cell type densities
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count
    use_label:         The cell type results to use in counting
    use_het:                The stoarge place for result
    distance: int           Distance to determine the neighbours (default is the nearest neighbour), distance=0 means within spot

    Returns
    -------
    adata: AnnData          With the counts of specified clusters in nearby spots stored as adata.uns['het']
    """

    library_id = list(adata.uns["spatial"].keys())[0]
    # between spot
    if distance != 0:
        # automatically calculate distance if not given, won't overwrite distance=0 which is within-spot
        if not distance:
            # calculate default neighbour distance
            scalefactors = next(iter(adata.uns["spatial"].values()))["scalefactors"]
            distance = (
                scalefactors["spot_diameter_fullres"]
                * scalefactors[
                    "tissue_"
                    + adata.uns["spatial"][library_id]["use_quality"]
                    + "_scalef"
                ]
                * 2
            )

        counts_ct = pd.DataFrame(0, adata.obs_names, ["CT"])

        # get neighbour spots for each spot
        coor = adata.obs[["imagerow", "imagecol"]]
        point_tree = spatial.cKDTree(coor)
        neighbours = []
        for spot in adata.obs_names:
            n_index = point_tree.query_ball_point(
                np.array(
                    [adata.obs["imagerow"].loc[spot], adata.obs["imagecol"].loc[spot]]
                ),
                distance,
            )
            neighbours = [item for item in adata.obs_names[n_index]]
            counts_ct.loc[spot] = (
                (adata.uns[use_label].loc[neighbours] > 0.2).sum() > 0
            ).sum()
        adata.obsm[use_het] = counts_ct["CT"].values

    # within spot
    else:
        # count the cell types with prob > 0.2 in the result of label transfer
        adata.obsm[use_het] = (adata.uns[use_label] > 0.2).sum(axis=1)

    if verbose:
        print(
            "Counts for cluster (cell type) diversity stored into adata.uns['"
            + use_het
            + "']"
        )

    return adata


def get_edges(adata: AnnData, L_bool: np.array, R_bool: np.array, sig_bool: np.array):
    """Gets a list edges representing significant interactions.

    Parameters
    ----------
    adata: AnnData
    L_bool: np.array<bool>  len(L_bool)==len(adata), True if ligand expressed in that spot.
    R_bool: np.array<bool>  len(R_bool)==len(adata), True if receptor expressed in that spot.
    sig_bool np.array<bool>:   len(sig_bool)==len(adata), True if spot has significant LR interactions.
    Returns
    -------
    edge_list_unique:   list<list<str>> Either a list of tuples (directed), or
                        list of sets (undirected), indicating unique significant
                        interactions between spots.
    """
    # Getting the neighbourhoods #
    neighbours, neighbourhood_bcs, neighbourhood_indices = get_neighbourhoods(adata)

    # Getting the edges to draw in-between #
    L_spot_indices = np.where(np.logical_and(L_bool, sig_bool))[0]
    R_spot_indices = np.where(np.logical_and(R_bool, sig_bool))[0]

    # To keep the get_between_spot_edge_array function happy #
    cell_data = np.ones((1, len(sig_bool)))[0, :].astype(np.float_)

    # Retrieving the edges #
    gene_bools = [R_bool, L_bool]
    edge_list = []
    for i, spot_indices in enumerate([L_spot_indices, R_spot_indices]):
        # Subsetting to the relevant neighbourhoods #
        neigh_bcs = List()
        neigh_indices = List()
        for j in spot_indices:
            neigh_bcs.append(neighbourhood_bcs[j])
            neigh_indices.append(neighbourhood_indices[j])

        # Getting the edges in this neighbourhood #
        edges = init_edge_list(neighbourhood_bcs)  # Note this has 1 pseudo edge
        if len(spot_indices) != 0:
            get_between_spot_edge_array(
                edges, neigh_bcs, neigh_indices, gene_bools[i], cell_data
            )
        edge_list.append(edges[1:])

    return edge_list


def count_interactions(
    adata,
    all_set,
    mix_mode,
    use_label,
    sig_bool,
    gene1_bool,
    gene2_bool,
    tissue_types=None,
    cell_type_props=None,
    cell_prop_cutoff=0.2,
    trans_dir=True,
):
    """Counts the interactions."""
    # Getting minimal information necessary for the counting #
    (
        spot_bcs,
        cell_data,
        neighbourhood_bcs,
        neighbourhood_indices,
    ) = get_data_for_counting(adata, use_label, mix_mode, all_set)

    # if trans_dir, rows are transmitter cell, cols receiver, otherwise reverse.
    int_matrix = np.zeros((len(all_set), len(all_set)), dtype=int)
    for i, cell_A in enumerate(all_set):  # transmitter if trans_dir else reciever
        # Determining which spots have cell type A #
        if not mix_mode:
            A_bool = tissue_types == cell_A
        else:
            col_A = [
                col for i, col in enumerate(cell_type_props.columns) if cell_A in col
            ][0]
            A_bool = cell_type_props.loc[:, col_A].values > cell_prop_cutoff

        A_gene1_bool = np.logical_and(A_bool, gene1_bool)
        A_gene1_sig_bool = np.logical_and(A_gene1_bool, sig_bool)
        A_gene1_sig_indices = np.where(A_gene1_sig_bool)[0]

        for j, cell_B in enumerate(all_set):  # receiver if trans_dir else transmitter
            cellA_cellB_counts = len(
                edge_core(
                    cell_data,
                    j,
                    neighbourhood_bcs,
                    neighbourhood_indices,
                    spot_indices=A_gene1_sig_indices,
                    neigh_bool=gene2_bool,
                    cutoff=cell_prop_cutoff,
                )
            )
            int_matrix[i, j] = cellA_cellB_counts

    return int_matrix if trans_dir else int_matrix.transpose()


def get_interaction_pvals(
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
):
    """ Perturbs the cell labels to get background count frequency to estimate \
        p-values.
    """

    # Counting how many times permutation of spots cell data creates interaction
    # counts greater than that observed, in order to calculate p-values.
    greater_counts = np.zeros(int_matrix.shape).astype(int)
    indices = np.array([i for i in range(cell_data.shape[0])])
    # If dealing with discrete data, no need to randomise columns indendently #
    discrete = np.all(np.logical_or(cell_data == 0, cell_data == 1))
    for i in range(n_perms):
        # Permuting the cell data by swapping between spots for each column #
        if not discrete:
            perm_data = cell_data.copy()
            for j in range(cell_data.shape[1]):
                rand_indices = np.random.choice(indices, cell_data.shape[0], False)
                perm_data[:, j] = cell_data[rand_indices, j]
        else:
            rand_indices = np.random.choice(indices, cell_data.shape[0], False)
            perm_data = cell_data[rand_indices, :]

        # Calculating interactions for permuted labels #
        perm_matrix = get_interaction_matrix(
            perm_data,
            neighbourhood_bcs,
            neighbourhood_indices,
            all_set,
            sig_bool,
            L_bool,
            R_bool,
            cell_prop_cutoff,
        ).astype(int)
        perm_greater = (perm_matrix >= int_matrix).astype(int)
        greater_counts += perm_greater

    # Calculating the pvalues #
    int_pvals = greater_counts / n_perms
    return int_pvals


@njit
def get_interaction_matrix(
    cell_data,
    neighbourhood_bcs,
    neighbourhood_indices,
    all_set,
    sig_bool,
    L_bool,
    R_bool,
    cell_prop_cutoff,
):
    """Gets the interaction count matrix."""
    # Now counting the interactions under 3 situations:
    # 1) sig spot with ligand, only neighbours with receptor relevant
    # 2) sig spot with receptor, only neighbours with ligand relevant
    # NOTE, A<->B is double counted, but on different side of matrix.
    # (if bidirectional interaction between two spots, counts as two seperate interactions).
    LR_edges = get_interactions(
        cell_data,
        neighbourhood_bcs,
        neighbourhood_indices,
        all_set,
        sig_bool,
        L_bool,
        R_bool,
        cell_prop_cutoff=cell_prop_cutoff,
        # sig ligand->receptor mode
    )
    RL_edges = get_interactions(
        cell_data,
        neighbourhood_bcs,
        neighbourhood_indices,
        all_set,
        sig_bool,
        R_bool,
        L_bool,
        cell_prop_cutoff=cell_prop_cutoff,
        # sig receptor->ligand mode
    )

    # Counting the number of unique interacting edges
    # between different cell type via indicate LR
    int_matrix = np.zeros((all_set.shape[0], all_set.shape[0]))
    edge_i = 0
    for i in range(all_set.shape[0]):
        for j in range(all_set.shape[0]):
            RL_Atrans_Bedges = LR_edges[edge_i]
            LR_Atrans_Bedges = RL_edges[edge_i]
            edge_i += 1
            max_len = max([len(RL_Atrans_Bedges), len(LR_Atrans_Bedges)])
            if max_len == 0:  # Nothing to count #
                continue

            edge_starts = List()
            edge_ends = List()
            for k in range(max_len):
                if k < len(RL_Atrans_Bedges):
                    edge_starts.append(RL_Atrans_Bedges[k][0])
                    edge_ends.append(RL_Atrans_Bedges[k][1])
                if k < len(LR_Atrans_Bedges):
                    edge_starts.append(LR_Atrans_Bedges[k][0])
                    edge_ends.append(LR_Atrans_Bedges[k][1])
            Atrans_Bedges = List()
            Atrans_Bedges.append((edge_starts[0], edge_ends[0]))  # for typing
            add_unique_edges(Atrans_Bedges, edge_starts, edge_ends)
            # Atrans_Bedges = np.unique(RL_Atrans_Bedges + LR_Atrans_Bedges)
            int_matrix[i, j] = len(Atrans_Bedges) - 1  # since added edge for type

    return int_matrix


@njit
def get_interactions(
    cell_data,
    neighbourhood_bcs,
    neighbourhood_indices,
    all_set,
    sig_bool,
    gene1_bool,
    gene2_bool,
    cell_prop_cutoff=None,
):
    """ Gets spot edges between cell types where the first cell type fits \
        criteria of gene1_bool, & second second cell type of gene2_bool.
    """

    # Creating list of lists to store edges for respective cell types #
    interaction_edges = List()

    # Now retrieving the interaction edges #
    for i in range(all_set.shape[0]):

        # Determining which spots have cell type A #
        A_bool_2 = cell_data[:, i] > cell_prop_cutoff
        A_gene1_bool = np.logical_and(A_bool_2, gene1_bool)

        A_gene1_sig_bool = np.logical_and(A_gene1_bool, sig_bool)
        n_true = A_gene1_sig_bool.sum()
        A_gene1_sig_indices = np.zeros((1, n_true), dtype=np.int_)[
            0, :
        ]  # np.where(A_gene1_sig_bool)[0]
        index = 0
        for k in range(A_gene1_sig_bool.shape[0]):
            if A_gene1_sig_bool[k]:
                A_gene1_sig_indices[index] = k
                index += 1

        for j in range(all_set.shape[0]):
            edge_list = edge_core(
                cell_data,
                j,
                neighbourhood_bcs,
                neighbourhood_indices,
                spot_indices=A_gene1_sig_indices,
                neigh_bool=gene2_bool,
                cutoff=cell_prop_cutoff,
            )

            interaction_edges.append(edge_list)

    return interaction_edges


def create_grids(adata: AnnData, num_row: int, num_col: int, radius: int = 1):
    """Generate screening grids across the tissue sample
    Parameters
    ----------
    adata: AnnData          The data object to generate grids on
    num_row: int            Number of rows
    num_col: int            Number of columns
    radius: int             Radius to determine neighbours (default: 1, nearest)

    Returns
    -------
    grids                 The individual grids defined by left and upper side
    width                   Width of grids
    height                  Height of grids
    """

    from itertools import chain

    coor = adata.obs[["imagerow", "imagecol"]]
    max_x = max(coor["imagecol"])
    min_x = min(coor["imagecol"])
    max_y = max(coor["imagerow"])
    min_y = min(coor["imagerow"])
    width = (max_x - min_x) / num_col
    height = (max_y - min_y) / num_row
    grids, neighbours = [], []
    # generate grids from top to bottom and left to right
    for n in range(num_row * num_col):
        neighbour = []
        x = min_x + n // num_row * width  # left side
        y = min_y + n % num_row * height  # upper side
        grids.append([x, y])

        # get neighbouring grids
        row = n % num_row
        col = n // num_row
        a = np.arange(num_row * num_col).reshape(num_col, num_row).T
        nb_matrix = [
            [
                a[i][j] if 0 <= i < a.shape[0] and 0 <= j < a.shape[1] else -1
                for j in range(col - radius, col + 1 + radius)
            ]
            for i in range(row - radius, row + 1 + radius)
        ]
        for item in nb_matrix:
            neighbour = chain(neighbour, item)
        neighbour = list(set(list(neighbour)))
        neighbours.append(
            [
                grid
                for grid in neighbour
                if not (grid == n and radius > 0) and grid != -1
            ]
        )

    return grids, width, height, neighbours


def count_grid(
    adata: AnnData,
    num_row: int = 30,
    num_col: int = 30,
    use_label: str = None,
    use_het: str = "cci_het_grid",
    radius: int = 1,
    verbose: bool = True,
) -> AnnData:
    """Count the cell type densities
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count
    num_row: int            Number of grids on height
    num_col: int            Number of grids on width
    use_label:         The cell type results to use in counting
    use_het:                The stoarge place for result
    radius: int             Distance to determine the neighbour grids (default: 1=nearest), radius=0 means within grid

    Returns
    -------
    adata: AnnData          With the counts of specified clusters in each grid of the tissue stored as adata.uns['het']
    """

    coor = adata.obs[["imagerow", "imagecol"]]
    grids, width, height, neighbours = create_grids(adata, num_row, num_col, radius)
    counts = pd.DataFrame(0, range(len(grids)), ["CT"])
    for n, grid in enumerate(grids):
        spots = coor[
            (coor["imagecol"] > grid[0])
            & (coor["imagecol"] < grid[0] + width)
            & (coor["imagerow"] < grid[1])
            & (coor["imagerow"] > grid[1] - height)
        ]
        counts.loc[n] = (adata.obsm[use_label].loc[spots.index] > 0.2).sum().sum()
    adata.obsm[use_het] = (counts / counts.max())["CT"]

    if verbose:
        print(
            "Counts for cluster (cell type) diversity stored into data.uns['"
            + use_het
            + "']"
        )

    return adata
