"""Spot-to-spot edges of significant LR interactions, for plotting."""

import numpy as np
from anndata import AnnData
from numba import njit
from numba.typed import List

from .neighbours import get_neighbourhoods


def get_edges(
    adata: AnnData,
    l_bool: np.ndarray,
    r_bool: np.ndarray,
    sig_bool: np.ndarray,
):
    """Gets a list edges representing significant interactions.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing spatial transcriptomics data.
    l_bool : np.ndarray of bool, shape (n_spots,)
        Boolean array indicating spots where the ligand is expressed.
    r_bool : np.ndarray of bool, shape (n_spots,)
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
    l_spot_indices = np.where(np.logical_and(l_bool, sig_bool))[0]
    r_spot_indices = np.where(np.logical_and(r_bool, sig_bool))[0]

    # To keep the get_between_spot_edge_array function happy #
    cell_data = np.ones((1, len(sig_bool)))[0, :].astype(np.float64)

    # Retrieving the edges #
    gene_bools = [r_bool, l_bool]
    edge_list = []
    for i, spot_indices in enumerate([l_spot_indices, r_spot_indices]):
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
                edges,
                neigh_bcs,
                neigh_indices,
                gene_bools[i],
                cell_data,
            )
        edge_list.append(edges[1:])

    return edge_list


@njit
def init_edge_list(neighbourhood_bcs):
    """ Initialises the edge-list in a way which ensures consistent typing to \
        prevent errors with Numba.
    """

    # Initialising the edge list #
    edge_list = List()
    # This ensures consistent typing #
    for i in range(len(neighbourhood_bcs)):
        if len(neighbourhood_bcs[i][1]) > 0:
            edge_list.append((neighbourhood_bcs[i][0], neighbourhood_bcs[i][1][0]))
            break
    return edge_list


@njit
def get_between_spot_edge_array(
    edge_list: List,
    neighbourhood_bcs: List,
    neighbourhood_indices: List,
    neigh_bool: np.ndarray,
    cell_data: np.ndarray,
    cutoff: float = 0,
):
    """ Populates edge_list with edges linking spots with a valid neighbour \
        of a given cell type. Validity of neighbour determined by neigh_bool, \
        which can indicate whether the neighbour expresses a certain ligand \
        or receptor. neighbourhood_bcs and neighbourhood_indices hold, per
        spot, (barcode, neighbour barcodes) and (index, neighbour indices);
        cell_data is spots x cell types, and a neighbour counts as a cell
        type when its value is above cutoff.
    """
    edge_starts = List()
    edge_ends = List()
    for i in range(len(neighbourhood_bcs)):
        bcs, indices = neighbourhood_bcs[i], neighbourhood_indices[i]
        spot_bc, neigh_bcs = bcs
        neigh_indices = indices[1]
        # Subset the neighbours to only those fitting indicated criteria #
        neigh_bcs = neigh_bcs[neigh_bool[neigh_indices]]
        neigh_indices = neigh_indices[neigh_bool[neigh_indices]]

        if len(neigh_indices) == 0:  # No cases where neighbours meet criteria
            continue  # Don't add any interactions for this neighbourhood

        # Note that can keep all by inputting cell_data with all 1's #
        interact_neigh_bool = cell_data[neigh_indices] > cutoff

        # Retrieving the barcodes of the interacting neighbours #
        interact_neigh_bcs = neigh_bcs[interact_neigh_bool]
        for interact_neigh_bc in interact_neigh_bcs:
            edge_starts.append(spot_bc)
            edge_ends.append(interact_neigh_bc)

    # Getting the unique edges #
    if len(edge_starts) > 0:
        add_unique_edges(edge_list, edge_starts, edge_ends)


@njit
def add_unique_edges(edge_list, edge_starts, edge_ends):
    """ Adds the unique edges to the given edge list. \
    Complicated in order to satisfy Numba compilation in no-python mode.
    """
    n_edges = len(edge_starts)

    # Adding the unique edges #
    edge_added = np.zeros((1, len(edge_starts)))[0, :] == 1
    for i in range(n_edges):
        if not edge_added[i]:
            edge_start, edge_end = edge_starts[i], edge_ends[i]
            edge_list.append((edge_start, edge_end))
            for j in range(i, n_edges):
                edge_startj, edge_endj = edge_starts[j], edge_ends[j]
                # Direction doesn't matter #
                if (edge_start == edge_startj and edge_end == edge_endj) or (
                    edge_end == edge_startj and edge_start == edge_endj
                ):
                    edge_added[j] = True
