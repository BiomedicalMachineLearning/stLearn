import numpy as np
import pandas as pd
import scipy as sc
from numba import njit, prange
from numba.typed import List
import scipy.spatial as spatial
from anndata import AnnData
from .het import create_grids


def lr(
    adata: AnnData,
    use_lr: str = "cci_lr",
    distance: float = None,
    verbose: bool = True,
    neighbours: list = None,
    fast: bool = True,
) -> AnnData:

    """Calculate the proportion of known ligand-receptor co-expression among the neighbouring spots or within spots
    Parameters
    ----------
    adata: AnnData          The data object to scan
    use_lr: str             object to keep the result (default: adata.uns['cci_lr'])
    distance: float         Distance to determine the neighbours (default: closest), distance=0 means within spot
    neighbours: list        List of the neighbours for each spot, if None then computed. Useful for speeding up function.
    fast: bool              Whether to use the fast implimentation or not.

    Returns
    -------
    adata: AnnData          The data object including the results
    """

    # automatically calculate distance if not given, won't overwrite distance=0 which is within-spot
    distance = calc_distance(adata, distance)

    # # expand the LR pairs list by swapping ligand-receptor positions
    lr_pairs = adata.uns["lr"].copy()
    spot_lr1 = get_spot_lrs(adata, lr_pairs=lr_pairs, lr_order=True)
    spot_lr2 = get_spot_lrs(adata, lr_pairs=lr_pairs, lr_order=False)
    if verbose:
        print("Altogether " + str(spot_lr1.shape[1]) + " valid L-R pairs")

    # get neighbour spots for each spot according to the specified distance
    if type(neighbours) == type(None):
        neighbours = calc_neighbours(adata, distance, index=fast)

    # Calculating the scores, can have either the fast or the pandas version #
    if fast:
        adata.obsm[use_lr] = lr_core(spot_lr1.values, spot_lr2.values, neighbours, 0)
    else:
        adata.obsm[use_lr] = lr_pandas(spot_lr1, spot_lr2, neighbours)

    if verbose:
        print(
            "L-R interactions with neighbours are counted and stored into adata.obsm['"
            + use_lr
            + "']"
        )

    # return adata


def calc_distance(adata: AnnData, distance: float):
    """Automatically calculate distance if not given, won't overwrite \
        distance=0 which is within-spot.
    Parameters
    ----------
    adata: AnnData          The data object to scan
    distance: float         Distance to determine the neighbours (default: closest), distance=0 means within spot

    Returns
    -------
    distance: float         The automatically calcualted distance (or inputted distance)
    """
    if not distance and distance != 0:
        # for arranged-spots
        scalefactors = next(iter(adata.uns["spatial"].values()))["scalefactors"]
        library_id = list(adata.uns["spatial"].keys())[0]
        distance = (
            scalefactors["spot_diameter_fullres"]
            * scalefactors[
                "tissue_" + adata.uns["spatial"][library_id]["use_quality"] + "_scalef"
            ]
            * 2
        )
    return distance


def get_lrs_scores(
    adata: AnnData,
    lrs: np.array,
    neighbours: np.array,
    het_vals: np.array,
    min_expr: float,
    filter_pairs: bool = True,
    spot_indices: np.array = None,
):
    """Gets the scores for the indicated set of LR pairs & the heterogeneity values.
    Parameters
    ----------
    adata: AnnData   See run() doc-string.
    lrs: np.array    See run() doc-string.
    neighbours: np.array    Array of arrays with indices specifying neighbours of each spot.
    het_vals: np.array      Cell heterogeneity counts per spot.
    min_expr: float         Minimum gene expression of either L or R for spot to be considered to have reasonable score.
    filter_pairs: bool      Whether to filter to valid pairs or not.
    spot_indices: np.array  Array of integers speci
    Returns
    -------
    lrs: np.array   lr pairs from the database in format ['L1_R1', 'LN_RN']
    """
    if type(spot_indices) == type(None):
        spot_indices = np.array(list(range(len(adata))), dtype=np.int_)

    spot_lr1s = get_spot_lrs(
        adata, lr_pairs=lrs, lr_order=True, filter_pairs=filter_pairs
    )
    spot_lr2s = get_spot_lrs(
        adata, lr_pairs=lrs, lr_order=False, filter_pairs=filter_pairs
    )
    if filter_pairs:
        lrs = np.array(
            [
                "_".join(spot_lr1s.columns.values[i : i + 2])
                for i in range(0, spot_lr1s.shape[1], 2)
            ]
        )

    # Calculating the lr_scores across spots for the inputted lrs #
    lr_scores = get_scores(
        spot_lr1s.values, spot_lr2s.values, neighbours, het_vals, min_expr, spot_indices
    )

    if filter_pairs:
        return lr_scores, lrs
    else:
        return lr_scores


def get_spot_lrs(
    adata: AnnData,
    lr_pairs: list,
    lr_order: bool,
    filter_pairs: bool = True,
):
    """
    Parameters
    ----------
    adata: AnnData         The adata object to scan
    lr_pairs: list         List of the lr pairs (e.g. ['L1_R1', 'L2_R2',...]
    lr_order: bool         Forward version of the spot lr pairs (L1_R1), False indicates reverse (R1_L1)
    filter_pairs: bool     Whether to filter the pairs or not (check if present before subsetting).

    Returns
    -------
    spot_lrs: pd.DataFrame   Spots*GeneOrder, in format l1, r1, ... ln, rn if lr_order True, else r1, l1, ... rn, ln
    """
    df = adata.to_df()
    pairs_rev = [f'{pair.split("_")[1]}_{pair.split("_")[0]}' for pair in lr_pairs]
    pairs_wRev = []
    for i in range(len(lr_pairs)):
        pairs_wRev.extend([lr_pairs[i], pairs_rev[i]])

    if filter_pairs:
        pairs_wRev = [
            lr
            for i, lr in enumerate(pairs_wRev)
            if lr.split("_")[0] in df.columns and lr.split("_")[1] in df.columns
        ]

    lr_cols = [pair.split("_")[int(lr_order == False)] for pair in pairs_wRev]
    spot_lrs = df[lr_cols]
    return spot_lrs


def calc_neighbours(
    adata: AnnData,
    distance: float = None,
    index: bool = True,
    verbose: bool = True,
) -> List:
    """Calculate the proportion of known ligand-receptor co-expression among the neighbouring spots or within spots
    Parameters
    ----------
    adata: AnnData          The data object to scan
    distance: float         Distance to determine the neighbours (default: closest), distance=0 means within spot
    index: bool             Indicates whether to return neighbours as indices to other spots or names of other spots.

    Returns
    -------
    neighbours: numba.typed.List          List of np.array's indicating neighbours by indices for each spot.
    """
    if verbose:
        print("Calculating neighbours...")
    # get neighbour spots for each spot according to the specified distance
    coor = adata.obs[["imagerow", "imagecol"]]
    point_tree = spatial.cKDTree(coor)
    neighbours = []
    for i, spot in enumerate(adata.obs_names):
        if distance == 0:
            neighbours.append(np.array([i if index else spot]))
        else:
            n_index = point_tree.query_ball_point(
                np.array(
                    [adata.obs["imagerow"].loc[spot], adata.obs["imagecol"].loc[spot]]
                ),
                distance,
            )
            if index:
                n_index = np.array(n_index, dtype=np.int_)
                neighbours.append(n_index[n_index != i])
            else:
                n_spots = adata.obs_names[n_index]
                neighbours.append(n_spots[n_spots != spot])

    typed_neighs = List()
    [typed_neighs.append(neigh) for neigh in neighbours]

    n_neighs = np.array([len(neigh) for neigh in neighbours])
    if verbose:
        print(
            f"{len(np.where(n_neighs==0)[0])} spots with no neighbours, "
            f"{int(np.median(n_neighs))} median spot neighbours."
        )

    if np.all(n_neighs == 0):
        raise Exception(
            "All spots have no neighbours at current distance,"
            " set distance to higher value, or distance=0 for "
            "within-spot mode."
        )
    return typed_neighs


@njit
def lr_core(
    spot_lr1: np.ndarray,
    spot_lr2: np.ndarray,
    neighbours: List,
    min_expr: float,
    spot_indices: np.array,
) -> np.ndarray:
    """Calculate the lr scores for each spot.
    Parameters
    ----------
    spot_lr1: np.ndarray          Spots*Ligands
    spot_lr2: np.ndarray          Spots*Receptors
    neighbours: numba.typed.List          List of np.array's indicating neighbours by indices for each spot.
    min_expr: float               Minimum expression for gene to be considered expressed.
    Returns
    -------
    lr_scores: numpy.ndarray   Cells*LR-scores.
    """
    # Calculating mean of lr2 expressions from neighbours of each spot
    nb_lr2 = np.zeros((len(spot_indices), spot_lr2.shape[1]), np.float64)
    for i in range(len(spot_indices)):
        spot_i = spot_indices[i]
        nb_expr = spot_lr2[neighbours[spot_i], :]
        if nb_expr.shape[0] != 0:  # Accounting for no neighbours
            nb_expr_mean = nb_expr.sum(axis=0) / nb_expr.shape[0]
        else:
            nb_expr_mean = nb_expr.sum(axis=0)
        nb_lr2[i, :] = nb_expr_mean

    scores = (
        spot_lr1[spot_indices, :] * (nb_lr2 > min_expr)
        + (spot_lr1[spot_indices, :] > min_expr) * nb_lr2
    )
    spot_lr = scores.sum(axis=1)
    return spot_lr / 2


def lr_pandas(
    spot_lr1: np.ndarray,
    spot_lr2: np.ndarray,
    neighbours: list,
) -> np.ndarray:
    """Calculate the lr scores for each spot.
    Parameters
    ----------
    spot_lr1: pd.DataFrame          Cells*Ligands
    spot_lr2: pd.DataFrame          Cells*Receptors
    neighbours: list       List of neighbours by indices for each spot.
    Returns
    -------
    lr_scores: numpy.ndarray   Cells*LR-scores.
    """
    # function to calculate mean of lr2 expression between neighbours or within spot (distance==0) for each spot
    def mean_lr2(x):
        # get lr2 expressions from the neighbour(s)
        n_spots = neighbours[spot_lr2.index.tolist().index(x.name)]
        nbs = spot_lr2.loc[n_spots, :]
        if nbs.shape[0] > 0:  # if neighbour exists
            return nbs.sum() / nbs.shape[0]
        else:
            return 0

    # mean of lr2 expressions from neighbours of each spot
    nb_lr2 = spot_lr2.apply(mean_lr2, axis=1)

    # check whether neighbours exist
    try:
        nb_lr2.shape[1]
    except:
        raise ValueError("No neighbours found within given distance.")

    # keep value of nb_lr2 only when lr1 is also expressed on the spots
    spot_lr = pd.DataFrame(
        spot_lr1.values * (nb_lr2.values > 0) + (spot_lr1.values > 0) * nb_lr2.values,
    ).sum(axis=1)
    return spot_lr.values / 2


@njit(parallel=True)
def get_scores(
    spot_lr1s: np.ndarray,
    spot_lr2s: np.ndarray,
    neighbours: List,
    het_vals: np.array,
    min_expr: float,
    spot_indices: np.array,
) -> np.array:
    """Calculates the scores.
    Parameters
    ----------
    spot_lr1s: np.ndarray   Spots*GeneOrder1, in format l1, r1, ... ln, rn
    spot_lr2s: np.ndarray   Spots*GeneOrder2, in format r1, l1, ... rn, ln
    het_vals:  np.ndarray   Spots*Het counts
    neighbours: numba.typed.List          List of np.array's indicating neighbours by indices for each spot.
    min_expr: float               Minimum expression for gene to be considered expressed.
    Returns
    -------
    spot_scores: np.ndarray   Spots*LR pair of the LR scores per spot.
    """
    spot_scores = np.zeros((len(spot_indices), spot_lr1s.shape[1] // 2), np.float64)
    for i in prange(0, spot_lr1s.shape[1] // 2):
        i_ = i * 2  # equivalent to range(0, spot_lr1s.shape[1], 2)
        spot_lr1, spot_lr2 = spot_lr1s[:, i_ : (i_ + 2)], spot_lr2s[:, i_ : (i_ + 2)]
        lr_scores = lr_core(spot_lr1, spot_lr2, neighbours, min_expr, spot_indices)
        # The merge scores #
        lr_scores = np.multiply(het_vals[spot_indices], lr_scores)
        spot_scores[:, i] = lr_scores
    return spot_scores


def lr_grid(
    adata: AnnData,
    num_row: int = 10,
    num_col: int = 10,
    use_lr: str = "cci_lr_grid",
    radius: int = 1,
    verbose: bool = True,
) -> AnnData:

    """Calculate the proportion of known ligand-receptor co-expression among the neighbouring grids or within each grid
    Parameters
    ----------
    adata: AnnData          The data object to scan
    num_row: int            Number of grids on height
    num_col: int            Number of grids on width
    use_lr: str             object to keep the result (default: adata.uns['cci_lr'])
    radius: int             Distance to determine the neighbour grids (default: 1=nearest), radius=0 means within grid

    Returns
    -------
    adata: AnnData          The data object with the cci_lr grid result updated
    """

    # prepare data as pd.dataframe
    df = adata.to_df()
    if not isinstance(df, pd.DataFrame):
        if sc.sparse.issparse(df):
            df = pd.DataFrame(
                df.toarray(), index=adata.obs_names, columns=adata.var_names
            )
        else:
            df = pd.DataFrame(df, index=adata.obs_names, columns=adata.var_names)

    # generate grids and prepare grid level dataframe
    grids, width, height, neighbours = create_grids(adata, num_row, num_col, radius)
    df_grid = pd.DataFrame(0, index=range(len(grids)), columns=df.columns)

    # transform spot level into grid level
    coor = adata.obs[["imagerow", "imagecol"]]
    for n, grid in enumerate(grids):
        spots = coor[
            (coor["imagecol"] > grid[0])
            & (coor["imagecol"] < grid[0] + width)
            & (coor["imagerow"] < grid[1])
            & (coor["imagerow"] > grid[1] - height)
        ]
        df_grid.loc[n] = df.loc[spots.index].sum()

    # expand the LR pairs list by swapping ligand-receptor positions
    lr_pairs = adata.uns["lr"].copy()
    lr_pairs += [item.split("_")[1] + "_" + item.split("_")[0] for item in lr_pairs]

    # filter out those LR pairs not expressing in the dataset
    lr1 = [item.split("_")[0] for item in lr_pairs]
    lr2 = [item.split("_")[1] for item in lr_pairs]
    avail = [
        i for i, x in enumerate(lr1) if lr1[i] in df.columns and lr2[i] in df.columns
    ]
    grid_lr1 = df_grid[[lr1[i] for i in avail]]
    grid_lr2 = df_grid[[lr2[i] for i in avail]]
    if verbose:
        print("Altogether " + str(len(avail)) + " valid L-R pairs")

    # function to calculate mean of lr2 expression between neighbours or within spot (distance==0) for each spot
    def mean_lr2(x):
        # get the neighbour(s)' lr2 expressions
        nbs = grid_lr2.loc[neighbours[df_grid.index.tolist().index(x.name)], :]
        if nbs.shape[0] > 0:  # if neighbour exists
            return nbs.sum() / nbs.shape[0]
        else:
            return 0

    # mean of lr2 expression among neighbour grids
    nb_lr2 = grid_lr2.apply(mean_lr2, axis=1)

    # keep value of nb_lr2 only when lr1 is also expressed on the grids
    grid_lr = pd.DataFrame(
        grid_lr1.values * (nb_lr2.values > 0) + (grid_lr1.values > 0) * nb_lr2.values,
        index=df_grid.index,
        columns=[lr_pairs[i] for i in avail],
    ).sum(axis=1)
    adata.obsm[use_lr] = grid_lr.values / 2

    if verbose:
        print(
            "L-R interactions with neighbours are counted and stored into adata.uns['"
            + use_lr
            + "']"
        )

    return adata
