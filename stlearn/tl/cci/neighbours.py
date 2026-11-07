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
    verbose: bool = True,
) -> scipy.sparse.csr_matrix:
    """Each spot's neighbours: the other spots within distance, or the spot
    itself when distance=0 (within-spot mode).

    Parameters
    ----------
    adata (AnnData):
        The data object to scan
    distance (float):
        Distance to determine the neighbours (default: closest), distance=0 means
        within spot
    verbose (bool):
        Display debugging information

    Returns
    -------
    neighbours (scipy.sparse.csr_matrix):
        Spots by spots: row i's indices are spot i's neighbours, in the order
        the tree finds them, as st.tl.cci.run stores in obsp["spot_neighbours"].

    """
    if verbose:
        print("Calculating neighbours...")

    n_spots = adata.n_obs
    if distance == 0:
        # Within-spot mode: each spot is its only neighbour.
        indices = np.arange(n_spots, dtype=np.int32)
        indptr = np.arange(n_spots + 1, dtype=np.int64)
    else:
        coor = adata.obs[["imagerow", "imagecol"]].to_numpy()
        # One query for all spots, across all CPUs.
        found = spatial.cKDTree(coor).query_ball_point(
            coor,
            distance,
            workers=-1,
            return_sorted=False,
        )
        # Each row's spots as found, the spot itself left out.
        lengths = np.fromiter(map(len, found), np.int64, count=n_spots)
        rows = np.repeat(np.arange(n_spots), lengths)
        flat = np.concatenate(found).astype(np.int32)
        other = flat != rows
        indices = flat[other]
        indptr = np.zeros(n_spots + 1, np.int64)
        np.cumsum(np.bincount(rows[other], minlength=n_spots), out=indptr[1:])

    graph = scipy.sparse.csr_matrix(
        (np.ones(len(indices), np.int8), indices, indptr),
        shape=(n_spots, n_spots),
    )
    n_neighs = np.diff(graph.indptr)

    if verbose:
        print(
            f"{np.count_nonzero(n_neighs == 0)} spots with no neighbours, "
            f"{int(np.median(n_neighs))} median spot neighbours.",
        )
    return graph


def get_neighbourhoods(adata):
    """st.tl.cci.run stores results in obsp["spot_neighbours"]:
    (barcode, neighbour barcodes) and (index, neighbour indices) per spot."""
    if "spot_neighbours" in adata.obsp:
        graph = scipy.sparse.csr_matrix(adata.obsp["spot_neighbours"])
        spot_bcs = adata.obs_names.to_numpy().astype(str)
        neighbourhood_bcs, neighbourhood_indices = List(), List()
        for i in range(graph.shape[0]):
            start, end = graph.indptr[i], graph.indptr[i + 1]
            neigh = graph.indices[start:end].astype(np.int64)
            neighbourhood_indices.append((i, neigh))
            neighbourhood_bcs.append((spot_bcs[i], spot_bcs[neigh]))
        return neighbourhood_bcs, neighbourhood_indices
    else:
        raise ValueError(
            "No neighbours stored in adata.obsp['spot_neighbours']: call st.tl.cci.run."
        )
