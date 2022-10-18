from typing import Optional, Union
from anndata import AnnData
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from ._weighting_matrix import (
    calculate_weight_matrix,
    impute_neighbour,
    _WEIGHTING_MATRIX,
    _PLATFORM,
)
import stlearn
import scipy
from ..._compat import Literal


def SME_impute0(
    adata: AnnData,
    use_data: str = "raw",
    weights: _WEIGHTING_MATRIX = "weights_matrix_all",
    platform: _PLATFORM = "Visium",
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    using spatial location (S), tissue morphological feature (M) and gene expression (E) information to impute missing
    values

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_data
        input data, can be `raw` counts or log transformed data
    weights
        weighting matrix for imputation.
        if `weights_matrix_all`, matrix combined all information from spatial location (S),
        tissue morphological feature (M) and gene expression (E)
        if `weights_matrix_pd_md`, matrix combined information from spatial location (S),
        tissue morphological feature (M)
    platform
        `Visium` or `Old_ST`
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Anndata
    """
    if use_data == "raw":
        if isinstance(adata.X, csr_matrix):
            count_embed = adata.X.toarray()
        elif isinstance(adata.X, np.ndarray):
            count_embed = adata.X
        elif isinstance(adata.X, pd.Dataframe):
            count_embed = adata.X.values
        else:
            raise ValueError(
                f"""\
                    {type(adata.X)} is not a valid type.
                    """
            )
    else:
        count_embed = adata.obsm[use_data]

    calculate_weight_matrix(adata, platform=platform)

    impute_neighbour(adata, count_embed=count_embed, weights=weights)

    imputed_data = adata.obsm["imputed_data"].astype(float)
    mask = count_embed != 0
    count_embed_ = count_embed.astype(float)
    count_embed_[count_embed_ == 0] = np.nan
    adjusted_count_matrix = np.nanmean(np.array([count_embed_, imputed_data]), axis=0)
    adjusted_count_matrix[mask] = count_embed[mask]

    key_added = use_data + "_SME_imputed"
    adata.obsm[key_added] = adjusted_count_matrix

    print("The data adjusted by SME is added to adata.obsm['" + key_added + "']")

    return adata if copy else None


_COPY = Literal["pseudo_spot_adata", "combined_adata"]


def pseudo_spot(
    adata: AnnData,
    tile_path: Union[Path, str] = Path("/tmp/tiles"),
    use_data: str = "raw",
    crop_size: int = "auto",
    platform: _PLATFORM = "Visium",
    weights: _WEIGHTING_MATRIX = "weights_matrix_all",
    copy: _COPY = "pseudo_spot_adata",
) -> Optional[AnnData]:
    """\
    using spatial location (S), tissue morphological feature (M) and gene expression (E) information to impute
    gap between spots and increase resolution for gene detection

    Parameters
    ----------
    adata
        Annotated data matrix.
    use_data
        Input data, can be `raw` counts, log transformed data or dimension reduced space(`X_pca` and `X_umap`)
    tile_path
        Path to save spot image tiles
    crop_size
        Size of tiles
        if `auto`, automatically detect crop size
    weights
        Weighting matrix for imputation.
        if `weights_matrix_all`, matrix combined all information from spatial location (S),
        tissue morphological feature (M) and gene expression (E)
        if `weights_matrix_pd_md`, matrix combined information from spatial location (S),
        tissue morphological feature (M)
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
    from sklearn.linear_model import LinearRegression
    import math

    if platform == "Visium":
        img_row = adata.obs["imagerow"]
        img_col = adata.obs["imagecol"]
        array_row = adata.obs["array_row"]
        array_col = adata.obs["array_col"]
        rate = 3
        obs_df_ = adata.obs[["array_row", "array_col"]].copy()
        obs_df_.loc[:, "array_row"] = obs_df_["array_row"].apply(lambda x: x - 2 / 3)
        obs_df = adata.obs[["array_row", "array_col"]].copy()
        obs_df.loc[:, "array_row"] = obs_df["array_row"].apply(lambda x: x + 2 / 3)
        obs_df = obs_df.append(obs_df_).reset_index()
        obs_df.drop_duplicates(subset=["array_row", "array_col"], keep="last")

    elif platform == "Old_ST":
        img_row = adata.obs["imagerow"]
        img_col = adata.obs["imagecol"]
        array_row = adata.obs_names.map(lambda x: x.split("x")[1])
        array_col = adata.obs_names.map(lambda x: x.split("x")[0])
        rate = 1.5
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

        obs_df = obs_df_left.append(
            [
                obs_df_right,
                obs_df_up,
                obs_df_down,
                obs_df_left_up,
                obs_df_right_up,
                obs_df_left_down,
                obs_df_right_down,
            ]
        ).reset_index()
        obs_df.drop_duplicates(subset=["array_row", "array_col"], keep="last")
    else:
        raise ValueError(
            f"""\
                {platform!r} does not support.
                """
        )

    reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)

    reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)

    obs_df.loc[:, "imagerow"] = (
        obs_df.loc[:, "array_row"] * reg_row.coef_ + reg_row.intercept_
    )
    obs_df.loc[:, "imagecol"] = (
        obs_df.loc[:, "array_col"] * reg_col.coef_ + reg_col.intercept_
    )

    impute_coor = obs_df[["imagecol", "imagerow"]]
    coor = adata.obs[["imagecol", "imagerow"]].append(impute_coor)

    point_tree = scipy.spatial.cKDTree(coor)
    n_neighbour = []
    unit = math.sqrt(reg_row.coef_**2 + reg_col.coef_**2)
    for i in range(len(impute_coor)):
        current_neighbour = point_tree.query_ball_point(
            impute_coor.values[i], round(unit)
        )

        current_neighbour = [x for x in current_neighbour if x < len(adata)]
        n_neighbour.append(len(current_neighbour))

    obs_df["n_neighbour"] = n_neighbour
    obs_df = obs_df.loc[obs_df["n_neighbour"] > 1, :].reset_index()

    obs_df.index = obs_df.index.map(lambda x: "Pseudo_Spot_" + str(x))

    impute_df = pd.DataFrame(0, index=obs_df.index, columns=adata.var_names)

    pseudo_spot_adata = AnnData(impute_df, obs=obs_df)
    pseudo_spot_adata.uns["spatial"] = adata.uns["spatial"]

    if crop_size == "auto":
        crop_size = round(unit / 2)

    stlearn.pp.tiling(pseudo_spot_adata, tile_path, crop_size=crop_size)

    stlearn.pp.extract_feature(pseudo_spot_adata)

    if use_data == "raw":
        if isinstance(adata.X, csr_matrix):
            count_embed = adata.X.toarray()
        elif isinstance(adata.X, np.ndarray):
            count_embed = adata.X
        elif isinstance(adata.X, pd.Dataframe):
            count_embed = adata.X.values
        else:
            print(f"{type(adata.X)} is not a valid type")
    else:
        count_embed = adata.obsm[use_data]

    calculate_weight_matrix(
        adata, pseudo_spot_adata, pseudo_spots=True, platform=platform
    )

    impute_neighbour(pseudo_spot_adata, count_embed=count_embed, weights=weights)

    assert pseudo_spot_adata.shape == pseudo_spot_adata.obsm["imputed_data"].shape

    pseudo_spot_adata.X = pseudo_spot_adata.obsm["imputed_data"]

    pseudo_spot_adata = pseudo_spot_adata[np.sum(pseudo_spot_adata.X, axis=1) > 0]

    print("Done")

    if copy == "pseudo_spot_adata":
        return pseudo_spot_adata
    else:
        return _merge(adata, pseudo_spot_adata)


def _merge(
    adata1: AnnData,
    adata2: AnnData,
    copy: bool = True,
) -> Optional[AnnData]:
    merged_df = adata1.to_df().append(adata2.to_df())
    merged_df_obs = adata1.obs.append(adata2.obs)
    merged_adata = AnnData(merged_df, obs=merged_df_obs)
    merged_adata.uns["spatial"] = adata1.uns["spatial"]

    return merged_adata if copy else None
