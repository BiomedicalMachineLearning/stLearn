import numpy as np
import pandas as pd
import scipy as sc
import scipy.spatial as spatial
from anndata import AnnData
from .het import create_grids


def lr(
    adata: AnnData,
    use_lr: str = "cci_lr",
    distance: float = None,
) -> AnnData:

    """Calculate the proportion of known ligand-receptor co-expression among the neighbouring spots or within spots
    Parameters
    ----------
    adata: AnnData          The data object to scan
    use_lr: str             object to keep the result (default: adata.uns['cci_lr'])
    distance: float         Distance to determine the neighbours (default: closest), distance=0 means within spot

    Returns
    -------
    adata: AnnData          The data object including the results
    """

    # automatically calculate distance if not given, won't overwrite distance=0 which is within-spot
    if not distance and distance != 0:
        # for arranged-spots
        scalefactors = next(iter(adata.uns["spatial"].values()))["scalefactors"]
        distance = (
            scalefactors["spot_diameter_fullres"]
            * scalefactors["tissue_" + adata.uns["spatial"]["use_quality"] + "_scalef"]
            * 2
        )

    df = adata.to_df()

    # expand the LR pairs list by swapping ligand-receptor positions
    lr_pairs = adata.uns["lr"].copy()
    lr_pairs += [item.split("_")[1] + "_" + item.split("_")[0] for item in lr_pairs]

    # get neighbour spots for each spot according to the specified distance
    coor = adata.obs[["imagerow", "imagecol"]]
    point_tree = spatial.cKDTree(coor)
    neighbours = []
    for spot in adata.obs_names:
        if distance == 0:
            neighbours.append([spot])
        else:
            n_index = point_tree.query_ball_point(
                np.array(
                    [adata.obs["imagerow"].loc[spot], adata.obs["imagecol"].loc[spot]]
                ),
                distance,
            )
            neighbours.append(
                [item for item in df.index[n_index] if not (item == spot)]
            )

    # filter out those LR pairs that do not exist in the dataset
    lr1 = [item.split("_")[0] for item in lr_pairs]
    lr2 = [item.split("_")[1] for item in lr_pairs]
    avail = [
        i for i, x in enumerate(lr1) if lr1[i] in df.columns and lr2[i] in df.columns
    ]
    spot_lr1 = df[[lr1[i] for i in avail]]
    spot_lr2 = df[[lr2[i] for i in avail]]
    print("Altogether " + str(len(avail)) + " valid L-R pairs")

    # function to calculate mean of lr2 expression between neighbours or within spot (distance==0) for each spot
    def mean_lr2(x):
        # get lr2 expressions from the neighbour(s)
        nbs = spot_lr2.loc[neighbours[df.index.tolist().index(x.name)], :]
        if nbs.shape[0] > 0:  # if neighbour exists
            return (nbs > 0).sum() / nbs.shape[0]
        else:
            return 0

    # mean of lr2 expressions from neighbours of each spot
    nb_lr2 = spot_lr2.apply(mean_lr2, axis=1)

    """
    ######################### need to avoid system error when distance is small and no neighbours selected #########################
    """

    # keep value of nb_lr2 only when lr1 is also expressed on the spots
    spot_lr = pd.DataFrame(
        spot_lr1.values * (nb_lr2.values > 0) + (spot_lr1.values > 0) * nb_lr2.values,
        index=df.index,
        columns=[lr_pairs[i] for i in avail],
    ).sum(axis=1)
    adata.uns[use_lr] = spot_lr

    print(
        "L-R interactions with neighbours are counted and stored into adata.uns['"
        + use_lr
        + "']"
    )

    return adata


def lr_grid(
    adata: AnnData,
    num_row: int = 10,
    num_col: int = 10,
    use_lr: str = "cci_lr_grid",
    radius: int = 1,
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
    adata.uns[use_lr] = grid_lr

    print(
        "L-R interactions with neighbours are counted and stored into adata.uns['"
        + use_lr
        + "']"
    )

    return adata
