import numpy as np
import pandas as pd
from anndata import AnnData
import scipy.spatial as spatial
from numba.typed import List

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

def count_core(adata: AnnData, use_label: str, neighbours: List,
               spot_indices: np.array = None, neigh_bool: np.array = None,
               label_set=None, spot_mixtures: bool = True, cutoff: float = 0.2,
               ) -> np.array:
    """Get the cell type counts per spot, if spot_mixtures is True & there is \
        per spot deconvolution results available, then counts within spot. \
        If cell type deconvolution results not present but use_label in \
        adata.obs, then does counts number of cell types in the neighbourhood.

        Parameters
        ----------
        spot_lr1: np.ndarray          Spots*Ligands
        Returns
        -------
        cell_counts: numpy.ndarray   Cells*Cell_Type_Counts.
    """
    # Ensuring compatibility with current way of adding label_transfer to object
    if use_label == "label_transfer" or use_label == "predictions":
        obs_key, uns_key = "predictions", "label_transfer"
    else:
        obs_key, uns_key = use_label, use_label

    # Setting label_set if not present
    if type(label_set) == type(None):
        label_set = np.unique(adata.obs.loc[:,obs_key].values)

    # Setting neigh_bool if not present, is used to filter which spots can be neighbours
    if type(neigh_bool) == type(None):
        neigh_bool = np.array([1]*len(adata))

    # Setting the spot indices to do the counting
    if type(spot_indices) == type(None):
        spot_indices = np.array(list(range(len(adata))))

    neigh_zip = zip(spot_indices, [neighbours[i] for i in spot_indices])

    # Mixture mode
    if spot_mixtures and uns_key in adata.uns:
        # Making sure the label_set in consistent format with columns of adata.uns
        cols = list(adata.uns[uns_key])
        col_set = np.array([col.split(lab)[0]+lab+col.split(lab)[-1]
                                          for col, lab in zip(label_set, cols)])

        # within-spot
        if np.all(np.array([i in neighs for i, neighs in neigh_zip])==1):

            counts = (adata.uns[uns_key].loc[:,col_set].values[spot_indices, :]
                                                           > cutoff).sum(axis=1)

        # between-spot
        else:
            counts = np.zeros((1,len(adata)))[0]
            for i, neighs in neigh_zip:
                neighs = neighs[ neigh_bool[neighs] ]
                cell_bool = adata.uns[uns_key].loc[:,col_set].values[neighs,:] > 0.2
                counts[i] = sum(sum(cell_bool))

    # Absolute mode
    else:
        counts = np.zeros((1, len(adata)))[0]
        for i, neighs in neigh_zip:
            neighs = neighs[ neigh_bool[neighs] ]
            neigh_cell_types = adata.obs.loc[:,obs_key].values[neighs]
            cell_counts = [len(np.where(neigh_cell_types==cell_type)[0])
                                                     for cell_type in label_set]
            counts[i] = sum(cell_counts)

    return counts

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
