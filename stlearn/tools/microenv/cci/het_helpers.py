"""
Helper functions for het.py; primarily counting help.
"""

import numpy as np
import numba
from numba import types
from numba.typed import List
from numba import njit, jit

@njit
def edge_core(
    cell_data: np.ndarray,
    cell_type_index: int,
    neighbourhood_bcs: List,
    neighbourhood_indices: List,
    spot_indices: np.array = None,
    neigh_bool: np.array = None,
    cutoff: float = 0.2,
) -> np.array:
    """Gets the edges which connect inputted spots to neighbours of a given cell type.

        Parameters
        ----------
        cell_data: np.ndarray          Spots*CellTypes; value indicates \
                                        proportion of spot due to a specific \
                                        cell type. Rows sum to 1; pure spots \
                                        or spot annotations have a single 1 \
                                        per row.

        cell_type_index: int            Column of cell_data that contains the \
                                        cell type of interest.

        neighbourhood_bcs: List         List of lists, inner list for each \
                                        spot. First element of inner list is \
                                        spot barcode, second element is array \
                                        of neighbourhood spot barcodes.

        neighbourhood_indices: List     Same structure as neighbourhood_bcs, \
                                        but first inner list element is index \
                                        of the spot, and second is array of \
                                        neighbour indices.

        spot_indices: np.array          Array of indices indicating which spots \
                                        we want edges associated with. Can be \
                                        used to subset to significant spots.

        neigh_bool: np.array            Array of booleans of length n-spots, \
                                        True indicates the spot is an allowed \
                                        neighbour. This is useful where we only \
                                        want edges to neighbours which express \
                                        an indicated ligand or receptor.

        cutoff: float                   Cutoff above which cell is considered \
                                        to be present within the spot, is \
                                        applied on cell_data, and thereby allows \
                                        a spot to be counted as having multiple \
                                        cell types.

        Returns
        -------
        edges: List   List of 2-tuples containing spot barcodes, indicating \
                        an edge between a spot in spot_indices and a neighbour \
                        where neigh_bool is True, and either the
    """

    # Subsetting to relevant cell types #
    cell_data = cell_data[:, cell_type_index]

    # Initialising the edge list #
    edge_list = init_edge_list(neighbourhood_bcs)

    # If spots have no neighbours, no counting !
    if len(edge_list) == 0:
        return edge_list
    elif len(spot_indices) == 0:
        return edge_list[1:]

    ### Within-spot mode
    # within-spot, will have only itself as a neighbour in this mode
    within_mode = edge_list[0][0] == edge_list[0][1]
    if within_mode:
        # Numba implimentation #
        for i in spot_indices:
            if neigh_bool[i] and cell_data[i] > cutoff:
                edge_list.append((neighbourhood_bcs[i][0], neighbourhood_bcs[i][1][0]))

    ### Between-spot mode
    else:
        # Subsetting the neighbourhoods to relevant spots #
        neighbourhood_bcs_sub = List()
        neighbourhood_indices_sub = List()
        for spot_i in spot_indices:
            neighbourhood_bcs_sub.append(neighbourhood_bcs[spot_i])
            neighbourhood_indices_sub.append(neighbourhood_indices[spot_i])

        # Number of unique edges will be the count of interactions.
        get_between_spot_edge_array(
            edge_list,
            neighbourhood_bcs_sub,
            neighbourhood_indices_sub,
            neigh_bool,
            cell_data,
            cutoff=cutoff,
        )

    return edge_list[1:]  # Removing the initial edge added for typing #


@njit
def init_edge_list(neighbourhood_bcs):
    """ Initialises the edge-list in a way which ensures consistent typing to \
        prevent errors with Numba.
    """

    # Initialising the edge list #
    edge_list = List()
    # This ensures consistent typing #
    # spots_have_neighbours = False
    # first_spot = -1
    for i in range(len(neighbourhood_bcs)):
        if len(neighbourhood_bcs[i][1]) > 0:
            edge_list.append((neighbourhood_bcs[i][0], neighbourhood_bcs[i][1][0]))
            # spots_have_neighbours = True
            # first_spot = i
            break
    return edge_list


@njit
def get_between_spot_edge_array(
    edge_list: List,
    neighbourhood_bcs: List,
    neighbourhood_indices: List,
    neigh_bool: np.array,
    cell_data: np.array,
    cutoff: float = 0,
):
    """ Populates edge_list with edges linking spots with a valid neighbour \
        of a given cell type. Validity of neighbour determined by neigh_bool, \
        which can indicate whether the neighbour expresses a certain ligand \
        or receptor. See edge_core for parameter information.
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


def get_data_for_counting(adata, use_label, mix_mode, all_set):
    """Retrieves the minimal information necessary to perform edge counting."""
    # First determining how the edge counting needs to be performed #
    # Ensuring compatibility with current way of adding label_transfer to object
    if use_label == "label_transfer" or use_label == "predictions":
        obs_key, uns_key = "predictions", "label_transfer"
    else:
        obs_key, uns_key = use_label, use_label

    # Getting the neighbourhoods #
    neighbours, neighbourhood_bcs, neighbourhood_indices = get_neighbourhoods(
                                                                          adata)

    # Getting the cell type information; if not mixtures then populate
    # matrix with one's indicating pure spots.
    if mix_mode:
        cell_props = adata.uns[uns_key]
        cols = cell_props.columns.values.astype(str)
        col_order = [
            np.where([cell_type in col for col in cols])[0][0] for cell_type in all_set
        ]
        cell_data = adata.uns[uns_key].iloc[:, col_order].values.astype(np.float64)
    else:
        cell_labels = adata.obs.loc[:, obs_key].values
        cell_data = np.zeros((len(cell_labels), len(all_set)), dtype=np.float64)
        for i, cell_type in enumerate(all_set):
            cell_data[:, i] = (
                (cell_labels == cell_type).astype(np.int_).astype(np.float64)
            )

    spot_bcs = adata.obs_names.values.astype(str)
    return spot_bcs, cell_data, neighbourhood_bcs, neighbourhood_indices

def get_data_for_counting_OLD(adata, use_label, mix_mode, all_set):
    """Retrieves the minimal information necessary to perform edge counting."""
    # First determining how the edge counting needs to be performed #
    # Ensuring compatibility with current way of adding label_transfer to object
    if use_label == "label_transfer" or use_label == "predictions":
        obs_key, uns_key = "predictions", "label_transfer"
    else:
        obs_key, uns_key = use_label, use_label

    # Getting the neighbourhoods #
    neighbours, neighbourhood_bcs, neighbourhood_indices = get_neighbourhoods(adata)

    # Getting the cell type information; if not mixtures then populate
    # matrix with one's indicating pure spots.
    if mix_mode:
        cell_props = adata.uns[uns_key]
        cols = cell_props.columns.values.astype(str)
        col_order = [
            np.where([cell_type in col for col in cols])[0][0] for cell_type in all_set
        ]
        cell_data = adata.uns[uns_key].iloc[:, col_order].values.astype(np.float64)
    else:
        cell_labels = adata.obs.loc[:, obs_key].values
        cell_data = np.zeros((len(cell_labels), len(all_set)), dtype=np.float64)
        for i, cell_type in enumerate(all_set):
            cell_data[:, i] = (
                (cell_labels == cell_type).astype(np.int_).astype(np.float64)
            )

    spot_bcs = adata.obs_names.values.astype(str)
    return spot_bcs, cell_data, neighbourhood_bcs, neighbourhood_indices

@njit
def get_neighbourhoods_FAST(spot_bcs: np.array, spot_neigh_bcs: np.ndarray,
                            n_spots: int, str_dtype: str,
                            neigh_indices: np.array, neigh_bcs: np.array):
    """Gets the neighbourhood information, njit compiled."""

    # Determining the neighbour spots used for significance testing #
    ### Some initialisation of the lists with correct types for complilation ###
    neighbours = List( [neigh_indices] )[1:]
    neighbourhood_bcs = List([ (spot_bcs[0], neigh_bcs) ])[1:]
    neighbourhood_indices = List([ (0, neigh_indices) ])[1:]

    #neighbours = List( numba.int64[:] )
    #neighbourhood_bcs = List((numba.int64, numba.int64[:]))
    #neighbourhood_indices = List( (types.unicode_type, types.unicode_type[:]) )

    for i in range(spot_neigh_bcs.shape[0]):
        neigh_bcs = np.array(spot_neigh_bcs[i, :][0].split(","))
        neigh_bcs = neigh_bcs[neigh_bcs != ""]
        neigh_bcs_sub = List()
        for neigh_bc in neigh_bcs:
            if neigh_bc in spot_bcs:
                neigh_bcs_sub.append( neigh_bc )
        neigh_bcs = np.full((len(neigh_bcs_sub)), fill_value='',
                                                                dtype=str_dtype)
        neigh_indices = np.full(len(neigh_bcs_sub), fill_value=int(0),
                                                                 dtype=np.int64)
        for i, neigh_bc in enumerate(neigh_bcs_sub):
            neigh_indices[i] = np.where(spot_bcs == neigh_bc)[0][0]
            neigh_bcs[i] = neigh_bc

        neighbours.append( neigh_indices )
        neighbourhood_indices.append( (i, neigh_indices) )
        neighbourhood_bcs.append( (spot_bcs[i], neigh_bcs) )

    return neighbours, neighbourhood_bcs, neighbourhood_indices

def get_neighbourhoods(adata):
    """Gets the neighbourhood information."""

    # Old stlearn version where didn't store neighbourhood barcodes, not good
    #   for anndata subsetting!!
    if not "spot_neigh_bcs" in adata.obsm:
        # Determining the neighbour spots used for significance testing #
        neighbours = List()
        for i in range(adata.obsm["spot_neighbours"].shape[0]):
            neighs = np.array(adata.obsm["spot_neighbours"].values[i, :][0].split(","))
            neighs = neighs[neighs != ""].astype(int)
            # DONE: store neigh_bcs to get below to work reliably
            #       after subsetting anndata #
            # neighs = neighs[neighs<adata.shape[0]] # Removing subsetted spots..
            neighbours.append(neighs)

        # Getting the neighbourhood information #
        neighbourhood_bcs = List()
        neighbourhood_indices = List()
        spot_bcs = adata.obs_names.values.astype(str)
        for spot_i in range(len(spot_bcs)):
            neighbourhood_indices.append((spot_i, neighbours[spot_i]))
            neighbourhood_bcs.append((spot_bcs[spot_i], spot_bcs[neighbours[spot_i]]))
    else:  # Newer version

        spot_bcs = adata.obs_names.values.astype(str)
        spot_neigh_bcs = adata.obsm["spot_neigh_bcs"].values

        max_len = 0
        for bc in spot_bcs:
            if len(bc) > max_len:
                max_len = len(bc)

        str_dtype = f"<U{max_len}"  # ensures correct typing

        n_spots = len(spot_bcs)
        neigh_indices = np.zeros((n_spots), np.int64)
        neigh_bcs = np.empty((n_spots), str_dtype)

        print(type(neigh_indices))
        print(type(neigh_bcs))

        return get_neighbourhoods_FAST(spot_bcs, spot_neigh_bcs,
                                       n_spots, str_dtype,
                                       neigh_indices, neigh_bcs)

        # Slow version
        # Determining the neighbour spots used for significance testing #
        # neighbours = List()
        # neighbourhood_bcs = List()
        # neighbourhood_indices = List()
        # spot_bcs = adata.obs_names.values.astype(str)
        # str_dtype = f"<U{max([len(bc) for bc in spot_bcs])}"  # ensures correct typing
        # for i in range(adata.shape[0]):
        #     neigh_bcs = np.array(
        #         adata.obsm["spot_neigh_bcs"].values[i, :][0].split(",")
        #     )
        #     neigh_bcs = neigh_bcs[neigh_bcs != ""]
        #     neigh_bcs = np.array(
        #         [neigh_bc for neigh_bc in neigh_bcs if neigh_bc in spot_bcs],
        #         dtype=str_dtype,
        #     )
        #     neigh_indices = np.array(
        #         [np.where(spot_bcs == neigh_bc)[0][0] for neigh_bc in neigh_bcs],
        #         dtype=np.int64,
        #     )
        #     neighbours.append(neigh_indices)
        #     neighbourhood_indices.append((i, neigh_indices))
        #     neighbourhood_bcs.append((spot_bcs[i], neigh_bcs))

    return neighbours, neighbourhood_bcs, neighbourhood_indices
