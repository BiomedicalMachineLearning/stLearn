"""Spot neighbourhoods: computing them, and reading back the ones st.tl.cci.run
stored in adata.obsm.
"""

import numpy as np
import scipy.sparse
import scipy.spatial as spatial
from anndata import AnnData
from numba.typed import List

from ...utils import _image_scalef


def calc_distance(adata: AnnData, distance: float | None) -> float:
    """Automatically calculate distance if not given, won't overwrite \
        distance=0 which is within-spot.
    Parameters
    ----------
    adata: AnnData
        The data object to scan
    distance: float
        Distance to determine the neighbours (default: closest), distance=0 means
        within spot

    Returns
    -------
    distance: float
        The automatically calculate distance (or inputted distance)
    """
    if not distance and distance != 0:
        # for arranged-spots
        scalefactors = next(iter(adata.uns["spatial"].values()))["scalefactors"]
        distance = scalefactors["spot_diameter_fullres"] * _image_scalef(adata) * 2
    return distance


def calc_neighbours(
    adata: AnnData,
    distance: float | None = None,
    index: bool = True,
    verbose: bool = True,
) -> List:
    """Each spot's neighbours: the other spots within distance, or the spot
    itself when distance=0 (within-spot mode).

    Parameters
    ----------
    adata (AnnData):
        The data object to scan
    distance (float):
        Distance to determine the neighbours (default: closest), distance=0 means
        within spot
    index (bool):
        Indicates whether to return neighbours as indices to other spots or names of
        other spots.
    verbose (bool):
        Display debugging information

    Returns
    -------
    neighbours (numba.typed.List):
        List of np.array's indicating neighbours by indices for each spot.

    Raises
    ------
    ValueError
        If no spot has a neighbour at this distance.
    """
    if verbose:
        print("Calculating neighbours...")
    # get neighbour spots for each spot according to the specified distance
    coor = adata.obs[["imagerow", "imagecol"]].to_numpy()
    point_tree = spatial.cKDTree(coor)
    # One query for all spots, across all CPUs.
    found = point_tree.query_ball_point(
        coor,
        distance,
        workers=-1,
        return_sorted=False,
    )
    neighbours = []
    for i, spot in enumerate(adata.obs_names):
        if distance == 0:
            neighbours.append(np.array([i if index else spot]))
        else:
            n_index = found[i]
            if index:
                n_index = np.array(n_index, dtype=np.int32)
                neighbours.append(n_index[n_index != i])
            else:
                n_spots = adata.obs_names[n_index]
                neighbours.append(n_spots[n_spots != spot])

    typed_neighs = List()
    [typed_neighs.append(neigh) for neigh in neighbours]

    n_neighs = np.array([len(neigh) for neigh in neighbours])
    if verbose:
        print(
            f"{len(np.where(n_neighs == 0)[0])} spots with no neighbours, "
            f"{int(np.median(n_neighs))} median spot neighbours.",
        )

    if np.all(n_neighs == 0):
        raise ValueError(
            "All spots have no neighbours at current distance,"
            " set distance to higher value, or distance=0 for "
            "within-spot mode.",
        )
    return typed_neighs


def neighbourhood_matrix(adata: AnnData, distance: float) -> scipy.sparse.csr_matrix:
    """Spots by spots; row i marks spot i's neighbourhood within distance, the
    spot itself included."""
    coor = adata.obs[["imagerow", "imagecol"]].to_numpy()
    found = spatial.cKDTree(coor).query_ball_point(coor, distance, workers=-1)
    rows = np.repeat(np.arange(len(found)), list(map(len, found)))
    return scipy.sparse.csr_matrix(
        (np.ones(len(rows), np.int32), (rows, np.concatenate(found))),
        shape=(len(found), len(found)),
    )


def neighbourhoods_from_barcodes(
    spot_bcs: np.ndarray,
    spot_neigh_bcs: np.ndarray,
    str_dtype: str,
):
    """(barcode, neighbour barcodes) and (index, neighbour indices) per spot, from
    the comma-joined neighbour barcodes st.tl.cci.run stores. Neighbours no longer
    in the data, after subsetting, are skipped.
    """
    neighbourhood_bcs, neighbourhood_indices = [], []
    spot_bc_to_index = {bc: i for i, bc in enumerate(spot_bcs)}

    for i in range(spot_neigh_bcs.shape[0]):
        neigh_bcs = np.array(spot_neigh_bcs[i, :][0].split(","))
        neigh_bcs = neigh_bcs[neigh_bcs != ""]

        neigh_bcs_array, neigh_indices = [], []
        for neigh_bc in neigh_bcs:
            if neigh_bc in spot_bc_to_index:
                neigh_bcs_array.append(neigh_bc)
                neigh_indices.append(spot_bc_to_index[neigh_bc])

        neigh_bcs_array = np.array(neigh_bcs_array, dtype=str_dtype)
        neigh_indices = np.array(neigh_indices, dtype=np.int64)

        neighbourhood_indices.append((i, neigh_indices))
        neighbourhood_bcs.append((spot_bcs[i], neigh_bcs_array))

    return List(neighbourhood_bcs), List(neighbourhood_indices)


def get_neighbourhoods(adata):
    """Gets the neighbourhood information stored by st.tl.cci.run."""

    # Old stlearn version where didn't store neighbourhood barcodes, not good
    #   for anndata subsetting!!
    if "spot_neigh_bcs" not in adata.obsm:
        # Determining the neighbour spots used for significance testing #
        neighbours = List()
        for i in range(adata.obsm["spot_neighbours"].shape[0]):
            neighs = np.array(adata.obsm["spot_neighbours"].values[i, :][0].split(","))
            neighs = neighs[neighs != ""].astype(int)
            neighbours.append(neighs)

        # Getting the neighbourhood information #
        neighbourhood_bcs = List()
        neighbourhood_indices = List()
        spot_bcs = adata.obs_names.to_numpy().astype(str)
        for spot_i in range(len(spot_bcs)):
            neighbourhood_indices.append((spot_i, neighbours[spot_i]))
            neighbourhood_bcs.append((spot_bcs[spot_i], spot_bcs[neighbours[spot_i]]))

        return neighbourhood_bcs, neighbourhood_indices
    else:  # Newer version
        spot_bcs = adata.obs_names.to_numpy().astype(str)
        spot_neigh_bcs = adata.obsm["spot_neigh_bcs"].to_numpy().astype(str)

        max_len = 0
        for bc in spot_bcs:
            if len(bc) > max_len:
                max_len = len(bc)

        str_dtype = f"<U{max_len}"  # ensures correct typing

        return neighbourhoods_from_barcodes(spot_bcs, spot_neigh_bcs, str_dtype)
