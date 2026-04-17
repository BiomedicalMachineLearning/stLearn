from collections.abc import Iterable

import numpy as np
import pandas as pd
import scipy.spatial as spatial
from anndata import AnnData
from numba import jit, njit, prange
from numba.typed import List

from stlearn.tl.cci.het_helpers import (
    edge_core,
    get_between_spot_edge_array,
    get_data_for_counting,
    get_neighbourhoods,
    init_edge_list,
)


def count(
    adata: AnnData,
    use_label: str | None = None,
    use_het: str = "cci_het",
    verbose: bool = True,
    distance: float | None = None,
) -> AnnData:
    """Count the cell type densities
    Parameters
    ----------
    adata: AnnData
        The data object including the cell types to count
    use_label:
        The cell type results to use in counting
    use_het:
        The storage place for result
    distance: int
        Distance to determine the neighbours (default is the nearest neighbour),
        distance=0 means within spot

    Returns
    -------
    adata: AnnData
        With the counts of specified clusters in nearby spots stored as
        adata.uns['het']
    """

    library_id = list(adata.uns["spatial"].keys())[0]
    # between spot
    if distance != 0:
        # automatically calculate distance if not given, won't overwrite distance=0
        # which is within-spot
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


def get_edges(
    adata: AnnData, L_bool: np.ndarray, R_bool: np.ndarray, sig_bool: np.ndarray
):
    """Gets a list edges representing significant interactions.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing spatial transcriptomics data.
    L_bool : np.ndarray of bool, shape (n_spots,)
        Boolean array indicating spots where the ligand is expressed.
    R_bool : np.ndarray of bool, shape (n_spots,)
        Boolean array indicating spots where the receptor is expressed.
    sig_bool : np.ndarray of bool, shape (n_spots,)
        Boolean array indicating spots with significant ligand-receptor interactions.

    Returns
    -------
    edge_list_unique:   list<list<str>> Either a list of tuples (directed), or
                        list of sets (undirected), indicating unique significant
                        interactions between spots.
    """
    # Getting the neighbourhoods #
    neighbourhood_bcs, neighbourhood_indices = get_neighbourhoods(adata)

    # Getting the edges to draw in-between #
    L_spot_indices = np.where(np.logical_and(L_bool, sig_bool))[0]
    R_spot_indices = np.where(np.logical_and(R_bool, sig_bool))[0]

    # To keep the get_between_spot_edge_array function happy #
    cell_data = np.ones((1, len(sig_bool)))[0, :].astype(np.float64)

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
    cell_data = get_data_for_counting(adata, use_label, mix_mode, all_set)
    neighbourhood_bcs, neighbourhood_indices = get_neighbourhoods(adata)

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
            cell_a_cell_b_counts = len(
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
            int_matrix[i, j] = cell_a_cell_b_counts

    return int_matrix if trans_dir else int_matrix.transpose()


@njit(parallel=True)
def get_interaction_pvals(
    int_matrix,
    n_perms,
    cell_data,
    neighbourhood_indices,
    all_set,
    sig_bool,
    L_bool,
    R_bool,
    cell_prop_cutoff,
):
    """Gets the p-values for the interaction counts."""

    # Counting how many times permutation of spots cell data creates interaction
    # counts greater than that observed, in order to calculate p-values.
    shape_ = (n_perms, int_matrix.shape[0], int_matrix.shape[1])
    # Storing the instances where the count is greater randomly for each perm.
    # Allows for embarassing parallelisation.
    greater_counts = np.zeros(shape_, dtype=np.int64)
    indices = np.zeros((cell_data.shape[0]), dtype=np.int64)
    for i in range(cell_data.shape[0]):
        indices[i] = i

    # If dealing with discrete data, no need to randomise columns indendently #
    discrete = np.all(np.logical_or(cell_data == 0, cell_data == 1))
    for i in prange(n_perms):
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
            neighbourhood_indices,
            all_set,
            sig_bool,
            L_bool,
            R_bool,
            cell_prop_cutoff,
        )
        for row in range(int_matrix.shape[0]):
            for col in range(int_matrix.shape[1]):
                greater_counts[i, row, col] = (
                    perm_matrix[row, col] >= int_matrix[row, col]
                )

    # Numba parallel sums axis 0 efficiently
    out = np.zeros((int_matrix.shape[0], int_matrix.shape[1]), dtype=np.float64)
    for i in range(n_perms):
        for row in range(int_matrix.shape[0]):
            for col in range(int_matrix.shape[1]):
                out[row, col] += greater_counts[i, row, col]
    int_pvals = out / n_perms
    return int_pvals


@njit
def get_interaction_matrix(
    cell_data,
    neighbourhood_indices,
    all_set,
    sig_bool,
    L_bool,
    R_bool,
    cell_prop_cutoff,
):
    """Gets the interaction matrix for a given cell data matrix."""

    n_spots = cell_data.shape[0]
    n_types = all_set.shape[0]
    int_matrix = np.zeros((n_types, n_types), dtype=np.int64)

    for t1 in range(n_types):
        for t2 in range(n_types):
            s = {np.int64(-1)}
            s.clear()

            for i in range(n_spots):
                if not sig_bool[i]:
                    continue
                if cell_data[i, t1] <= cell_prop_cutoff:
                    continue

                neighs = neighbourhood_indices[i][1]

                for k in range(len(neighs)):
                    n_idx = neighs[k]
                    if cell_data[n_idx, t2] > cell_prop_cutoff:
                        valid = False
                        if L_bool[i] and R_bool[n_idx]:
                            valid = True
                        if R_bool[i] and L_bool[n_idx]:
                            valid = True

                        if valid:
                            u = np.int64(i)
                            v = np.int64(n_idx)
                            if u > v:
                                tmp = u
                                u = v
                                v = tmp
                            s.add((u << 32) | v)

            int_matrix[t1, t2] = len(s)

    return int_matrix


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
        neighbour: Iterable[float] = []
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
    use_label: str | None = None,
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
    use_label:              The cell type results to use in counting
    use_het:                The storage place for result
    radius: int             Distance to determine the neighbour grids
                            (default: 1=nearest), radius=0 means within grid

    Returns
    -------
    adata (AnnData):        With the counts of specified clusters in each grid of the
                            tissue stored as adata.uns['het']
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


@jit(parallel=True, forceobj=True)
def grid_parallel(
    grid_coords: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    n_row: int,
    n_col: int,
    xs: np.ndarray,
    ys: np.ndarray,
    cell_bcs: np.ndarray,
    grid_cell_counts: np.ndarray,
    grid_expr: np.ndarray,
    cell_expr: np.ndarray,
    use_label_bool: bool,
    cell_labels: np.ndarray,
    cell_info: np.ndarray,
    cell_set: np.ndarray,
):
    """Grids the gene expression information."""
    # generate grids from top to bottom and left to right
    for i in prange(n_col):
        x_left, x_right = xedges[i], xedges[i + 1]
        for j in range(n_row):
            n = (i * n_row) + j

            y_down, y_up = yedges[j], yedges[j + 1]
            grid_coords[n, :] = [(x_right + x_left) / 2, (y_up + y_down) / 2]

            # Now determining the cells within the gridded area #
            if i != n_col - 1 and j == n_row - 1:  # top left corner
                x_true = (xs >= x_left) & (xs < x_right)
                y_true = (ys <= y_up) & (ys > y_down)
            elif i == n_col - 1 and j != n_row - 1:  # bottom right corner
                x_true = (xs > x_left) & (xs <= x_right)
                y_true = (ys < y_up) & (ys >= y_down)
            else:  # average case
                x_true = (xs >= x_left) & (xs < x_right)
                y_true = (ys < y_up) & (ys >= y_down)

            cell_bool = x_true & y_true
            grid_cells = cell_bcs[cell_bool]
            # grid_cells_str = ",".join( grid_cells )
            # grid_bcs.append( grid_cells_str )
            grid_cell_counts[n] = len(grid_cells)
            # gridded_cells.extend( grid_cells )
            # cell_grid.extend( [f"grid_{n}"] * len(grid_cells) )

            # Summing the expression across these cells to get the grid expression #
            if len(grid_cells) > 0:
                grid_expr[n, :] = cell_expr[cell_bool, :].sum(axis=0)

            # If we have cell type information, will record #
            if use_label_bool and len(grid_cells) > 0:
                grid_cell_types = cell_labels[cell_bool]
                cell_info[n, :] = [
                    len(np.where(grid_cell_types == ct)[0]) / len(grid_cell_types)
                    for ct in cell_set
                ]
