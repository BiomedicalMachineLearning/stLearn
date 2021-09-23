"""
Helper functions for het.py; primarily counting help.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
import scipy.spatial as spatial
from numba.typed import List
from numba import njit, jit

@njit
def edge_core(cell_data: np.ndarray, cell_type_index: int,
               neighbourhood_bcs: List, neighbourhood_indices: List,
               spot_indices: np.array = None, neigh_bool: np.array = None,
               cutoff: float = 0.2,
               ) -> np.array:
    """Get the cell type counts per spot, if spot_mixtures is True & there is \
        per spot deconvolution results available, then counts within spot. \
        If cell type deconvolution results not present but use_label in \
        adata.obs, then counts number of cell types in the neighbourhood.

        Parameters
        ----------
        spot_lr1: np.ndarray          Spots*Ligands
        Returns
        -------
        counts: int   Total number of interactions satisfying the conditions, \
                      or np.array<set> if return_edges=True, where each set is \
                      an edge, only returns unique edges.
    """

    # Subsetting to relevant cell types #
    cell_data = cell_data[:, cell_type_index]

    # Initialising the edge list #
    edge_list = List()
    # This ensures consistent typing #
    spots_have_neighbours = False
    first_spot = -1
    for i in range(len(neighbourhood_bcs)):
        if len(neighbourhood_bcs[i][1]) > 0:
            edge_list.append( (neighbourhood_bcs[i][0],
                               neighbourhood_bcs[i][1][0]) )
            spots_have_neighbours = True
            first_spot = i
            break

    # If spots have no neighbours, no counting !
    if not spots_have_neighbours:
        return edge_list
    elif len(spot_indices)==0:
        return edge_list[1:]

    ### Within-spot mode
    # within-spot, will have only itself as a neighbour in this mode
    within_mode = neighbourhood_indices[first_spot][0] in \
                                            neighbourhood_indices[first_spot][1]
    if within_mode:
            # Since each edge link to the spot itself,
            # then need to count the number of significant spots where
            # cellA & cellB > cutoff, & the L/R are expressed.
            ## Getting spots where L/R expressed & cellA > cutoff
            # spots = [i in spot_indices and neigh_bool_
            #          for i, neigh_bool_ in enumerate(neigh_bool)]
            # ## For the spots where L/R expressed & cellA > cutoff, counting
            # ## how many have cellB > cutoff.
            # counts = (cell_data[spots]>cutoff).sum()
            # interact_indices = np.where(counts > 0)[0]
            # edge_list = [(spot_bcs[index]) for index in interact_indices]

            # Numba implimentation #
            for i in spot_indices:
                if neigh_bool[i] and cell_data[i] > cutoff:
                    edge_list.append( (neighbourhood_bcs[i][0],
                                       neighbourhood_bcs[i][1][0]) )

    ### Between-spot mode
    else:
        # Subsetting the neighbourhoods to relevant spots #
        neighbourhood_bcs_sub = List()
        neighbourhood_indices_sub = List()
        for spot_i in spot_indices:
            neighbourhood_bcs_sub.append( neighbourhood_bcs[spot_i] )
            neighbourhood_indices_sub.append( neighbourhood_indices[spot_i] )

        # Number of unique edges will be the count of interactions.
        get_between_spot_edge_array(edge_list, neighbourhood_bcs_sub,
                                                     neighbourhood_indices_sub,
                                                    neigh_bool, True,
                                             cell_data=cell_data, cutoff=cutoff)

    return edge_list[1:] # Removing the initial edge added for typing #

@njit
def get_between_spot_edge_array(edge_list: List, neighbourhood_bcs: List,
                                neighbourhood_indices: List,
                                neigh_bool: np.array,
                                count_cell_types: bool,
                                cell_data: np.array=None,
                                cutoff: float=0):
    """ undirected=False uses list instead of set to store edges,
    thereby giving direction.
    cell_data is either labels or label transfer scores.
    """
    edge_starts = List()
    edge_ends = List()
    #for bcs, indices in neigh_zip: #bc is cell barcode
    for i in range(len(neighbourhood_bcs)):
        bcs, indices = neighbourhood_bcs[i], neighbourhood_indices[i]
        spot_bc, neigh_bcs = bcs
        neigh_indices = indices[1]
        # Subset the neighbours to only those fitting indicated criteria #
        neigh_bcs = neigh_bcs[neigh_bool[neigh_indices]]
        neigh_indices = neigh_indices[neigh_bool[neigh_indices]]

        if len(neigh_indices) == 0: # No cases where neighbours meet criteria
            continue # Don't add any interactions for this neighbourhood

        # If we have cell data, need to subset neighbours meeting criteria
        if count_cell_types: # User needs to have input cell_data
            # If cutoff specified, then means cell_data refers to cell proportions
            #if mix_mode: # Inputted mixture data, user should have specific cutoff.
            # NOTE is always in mix_mode, for pure cell types just use 0s & 1s #
            interact_neigh_bool = cell_data[neigh_indices] > cutoff
            # interact_neigh_bool = interact_bool.sum(axis=1)
            # interact_neigh_bool = interact_neigh_bool == cell_data.shape[1]

        else: # Keep all neighbours with L | R as interacting
            interact_neigh_bool = np.ones((1,neigh_indices.shape[0]))[0,:]==1

        # Retrieving the barcodes of the interacting neighbours #
        interact_neigh_bcs = neigh_bcs[ interact_neigh_bool ]
        for interact_neigh_bc in interact_neigh_bcs:
            edge_starts.append( spot_bc )
            edge_ends.append( interact_neigh_bc )

    # Getting the unique edges #
    add_unique_edges(edge_list, edge_starts, edge_ends)

@njit
def add_unique_edges(edge_list, edge_starts, edge_ends):
    """ Adds the unique edges to the given edge list.
    """
    n_edges = len(edge_starts)

    # Adding the unique edges #
    edge_added = np.zeros((1,len(edge_starts)))[0,:]==1
    for i in range(n_edges):
        if not edge_added[i]:
            edge_start, edge_end = edge_starts[i], edge_ends[i]
            edge_list.append( (edge_start, edge_end) )
            for j in range(i, n_edges):
                edge_startj, edge_endj = edge_starts[j], edge_ends[j]
                # Direction doesn't matter #
                if (edge_start == edge_startj and edge_end == edge_endj) or \
                   (edge_end == edge_startj and edge_start == edge_endj):
                    edge_added[j] = True

def get_data_for_counting(adata, use_label, mix_mode, neighbours, all_set):
    """ Retrieves the minimal information necessary to perform edge counting.
    """
    # First determining how the edge counting needs to be performed #
    # Ensuring compatibility with current way of adding label_transfer to object
    if use_label == "label_transfer" or use_label == "predictions":
        obs_key, uns_key = "predictions", "label_transfer"
    else:
        obs_key, uns_key = use_label, use_label

    # Getting the neighbourhood information #
    neighbourhood_bcs = List()
    neighbourhood_indices = List()
    spot_bcs = adata.obs_names.values.astype(str)
    for spot_i in range(len(spot_bcs)):
        neighbourhood_indices.append( (spot_i, neighbours[spot_i]) )
        neighbourhood_bcs.append( (spot_bcs[spot_i],
                                   spot_bcs[neighbours[spot_i]]) )

    # Getting the cell type information; if not mixtures then populate
    # matrix with one's indicating pure spots.
    if mix_mode:
        cell_props = adata.uns[uns_key]
        cols = cell_props.columns.values.astype(str)
        col_order = [np.where([cell_type in col for col in cols])[0][0]
                                                       for cell_type in all_set]
        cell_data = adata.uns[uns_key].iloc[:, col_order].values.astype(np.float64)
    else:
        cell_labels = adata.obs.loc[:, obs_key].values
        cell_data = np.zeros( (len(cell_labels), len(all_set)), dtype=np.float64)
        for i, cell_type in enumerate(all_set):
            cell_data[:,i] = (cell_labels==cell_type).astype(np.int_)\
                                                             .astype(np.float64)

    return spot_bcs, cell_data, neighbourhood_bcs, neighbourhood_indices









