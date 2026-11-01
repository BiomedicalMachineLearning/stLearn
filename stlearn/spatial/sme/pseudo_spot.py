from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scipy
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse, vstack

import stlearn

from ._adjust import expression
from ._weighting_matrix import (
    _PLATFORM,
    _WEIGHTING_MATRIX,
    pseudo_spot_weights,
    row_col_by_platform,
)

_COPY = Literal["pseudo_spot_adata", "combined_adata"]


def pseudo_spot(
    adata: AnnData,
    tile_path: Path | str = Path("/tmp/tiles"),
    use_data: str = "raw",
    crop_size: str | int = "auto",
    platform: _PLATFORM = "Visium",
    weights: _WEIGHTING_MATRIX = "weights_matrix_all",
    copy: _COPY = "pseudo_spot_adata",
) -> AnnData | None:
    """\
    Improve spatial resolution by imputing (creating) new spots from existing ones
    using spatial, morphological, and expression (SME) information.

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_data
        Input data, can be `raw` counts, log transformed data or dimension
        reduced space(`X_pca` and `X_umap`)
    tile_path
        Path to save spot image tiles
    crop_size
        Size of tiles
        if `auto`, automatically detect crop size
    weights
        Weighting matrix for imputation.
        if `weights_matrix_all`, matrix combined all information from spatial
        location (S), tissue morphological feature (M) and gene expression (E)
        if `weights_matrix_pd_md`, matrix combined information from spatial
        location (S), tissue morphological feature (M)
    platform
        `Visium` or `Old_ST`
    copy
        Return Anndata
        if `pseudo_spot_adata`, imputed Anndata
        if `combined_adata`, merged Anndata of original data imputed Anndata.
    Returns
    -------
    Anndata
    """
    import math

    if platform == "Visium":
        obs_df_ = adata.obs[["array_row", "array_col"]].astype(np.float64)
        obs_df_.loc[:, "array_row"] = obs_df_["array_row"].apply(lambda x: x - 2 / 3)
        obs_df = adata.obs[["array_row", "array_col"]].astype(np.float64)
        obs_df.loc[:, "array_row"] = obs_df["array_row"].apply(lambda x: x + 2 / 3)
        obs_df = pd.concat([obs_df, obs_df_]).reset_index()
        obs_df = obs_df.drop_duplicates(subset=["array_row", "array_col"], keep="last")

    elif platform == "Old_ST":
        array_row = adata.obs_names.map(lambda x: float(x.split("x")[1]))
        array_col = adata.obs_names.map(lambda x: float(x.split("x")[0]))
        obs_df_left = pd.DataFrame(
            {"array_row": array_row.to_list(), "array_col": array_col.to_list()},
            dtype=np.float64,
        )
        obs_df_left.loc[:, "array_row"] = obs_df_left["array_row"].apply(
            lambda x: x - 1 / 2
        )

        obs_df_right = pd.DataFrame(
            {"array_row": array_row.to_list(), "array_col": array_col.to_list()},
            dtype=np.float64,
        )
        obs_df_right.loc[:, "array_row"] = obs_df_right["array_row"].apply(
            lambda x: x + 1 / 2
        )

        obs_df_up = pd.DataFrame(
            {"array_row": array_row.to_list(), "array_col": array_col.to_list()},
            dtype=np.float64,
        )
        obs_df_up.loc[:, "array_col"] = obs_df_up["array_col"].apply(
            lambda x: x - 1 / 2
        )

        obs_df_down = pd.DataFrame(
            {"array_row": array_row.to_list(), "array_col": array_col.to_list()},
            dtype=np.float64,
        )
        obs_df_down.loc[:, "array_col"] = obs_df_down["array_col"].apply(
            lambda x: x + 1 / 2
        )

        obs_df_left_up = pd.DataFrame(
            {"array_row": array_row.to_list(), "array_col": array_col.to_list()},
            dtype=np.float64,
        )
        obs_df_left_up.loc[:, "array_row"] = obs_df_left_up["array_row"].apply(
            lambda x: x - 1 / 2
        )
        obs_df_left_up.loc[:, "array_col"] = obs_df_left_up["array_col"].apply(
            lambda x: x - 1 / 2
        )

        obs_df_right_up = pd.DataFrame(
            {"array_row": array_row.to_list(), "array_col": array_col.to_list()},
            dtype=np.float64,
        )
        obs_df_right_up.loc[:, "array_row"] = obs_df_right_up["array_row"].apply(
            lambda x: x + 1 / 2
        )
        obs_df_right_up.loc[:, "array_col"] = obs_df_right_up["array_col"].apply(
            lambda x: x - 1 / 2
        )

        obs_df_left_down = pd.DataFrame(
            {"array_row": array_row.to_list(), "array_col": array_col.to_list()},
            dtype=np.float64,
        )
        obs_df_left_down.loc[:, "array_row"] = obs_df_left_down["array_row"].apply(
            lambda x: x - 1 / 2
        )
        obs_df_left_down.loc[:, "array_col"] = obs_df_left_down["array_col"].apply(
            lambda x: x + 1 / 2
        )

        obs_df_right_down = pd.DataFrame(
            {"array_row": array_row.to_list(), "array_col": array_col.to_list()},
            dtype=np.float64,
        )
        obs_df_right_down.loc[:, "array_row"] = obs_df_right_down["array_row"].apply(
            lambda x: x + 1 / 2
        )
        obs_df_right_down.loc[:, "array_col"] = obs_df_right_down["array_col"].apply(
            lambda x: x + 1 / 2
        )

        obs_df = pd.concat(
            [
                obs_df_left,
                obs_df_right,
                obs_df_up,
                obs_df_down,
                obs_df_left_up,
                obs_df_right_up,
                obs_df_left_down,
                obs_df_right_down,
            ]
        ).reset_index()
        obs_df = obs_df.drop_duplicates(subset=["array_row", "array_col"], keep="last")
    else:
        raise ValueError(f"""\
                {platform!r} does not support.
                """)

    reg_col, reg_row, _ = row_col_by_platform(adata, platform)
    # Spots without an array position (NaN) give no pseudo-spots: there is
    # nowhere to put them.
    obs_df = obs_df.dropna(subset=["array_row", "array_col"])

    obs_df.loc[:, "imagerow"] = (
        obs_df.loc[:, "array_row"] * reg_row.coef_[0] + reg_row.intercept_
    )
    obs_df.loc[:, "imagecol"] = (
        obs_df.loc[:, "array_col"] * reg_col.coef_[0] + reg_col.intercept_
    )

    impute_coor = obs_df[["imagecol", "imagerow"]]
    coor = pd.concat([adata.obs[["imagecol", "imagerow"]], impute_coor])

    point_tree = scipy.spatial.cKDTree(coor)
    n_neighbour = []
    unit = math.sqrt(reg_row.coef_[0] ** 2 + reg_col.coef_[0] ** 2)
    for i in range(len(impute_coor)):
        current_neighbour = point_tree.query_ball_point(
            impute_coor.values[i], round(unit)
        )

        current_neighbour = [x for x in current_neighbour if x < len(adata)]
        n_neighbour.append(len(current_neighbour))

    obs_df["n_neighbour"] = n_neighbour
    obs_df = obs_df.loc[obs_df["n_neighbour"] > 1, :].reset_index()

    obs_df.index = obs_df.index.map(lambda x: "Pseudo_Spot_" + str(x))

    # Empty (sparse) expression until it is imputed below.
    pseudo_spot_adata = AnnData(
        csr_matrix((len(obs_df), adata.n_vars)),
        obs=obs_df,
        var=pd.DataFrame(index=adata.var_names),
    )
    pseudo_spot_adata.uns["spatial"] = adata.uns["spatial"]

    actual_crop_size: int
    if crop_size == "auto":
        actual_crop_size = round(unit / 2)
    elif isinstance(crop_size, int):
        actual_crop_size = crop_size
    else:
        raise ValueError(f"crop_size must be 'auto' or an integer, got {crop_size}")
    stlearn.pp.tiling(pseudo_spot_adata, tile_path, crop_size=actual_crop_size)

    stlearn.pp.extract_feature(pseudo_spot_adata)

    count_embed = expression(adata, use_data)

    neighbours, top_weights = pseudo_spot_weights(
        adata, pseudo_spot_adata, weights=weights, platform=platform
    )
    pseudo_spot_adata.X = neighbours @ count_embed
    pseudo_spot_adata.obsm["top_weights"] = top_weights

    expressed = np.asarray(pseudo_spot_adata.X.sum(axis=1)).ravel() > 0
    pseudo_spot_adata = pseudo_spot_adata[expressed]

    print("Done")

    if copy == "pseudo_spot_adata":
        return pseudo_spot_adata
    else:
        return _merge(adata, pseudo_spot_adata)


def _merge(
    adata1: AnnData,
    adata2: AnnData,
    copy: bool = True,
) -> AnnData | None:
    # Stacked as they are: sparse expression stays sparse.
    if issparse(adata1.X) or issparse(adata2.X):
        merged_X = vstack([csr_matrix(adata1.X), csr_matrix(adata2.X)], format="csr")
    else:
        merged_X = np.vstack([adata1.X, adata2.X])

    merged_adata = AnnData(
        merged_X,
        obs=pd.concat([adata1.obs, adata2.obs]),
        var=pd.DataFrame(index=adata1.var_names),
    )
    merged_adata.uns["spatial"] = adata1.uns["spatial"]

    return merged_adata if copy else None
