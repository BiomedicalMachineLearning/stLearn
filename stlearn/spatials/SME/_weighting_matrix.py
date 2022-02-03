from sklearn.metrics import pairwise_distances
from typing import Optional, Union
from anndata import AnnData
import numpy as np
from ..._compat import Literal
from tqdm import tqdm

_PLATFORM = Literal["Visium", "Old_ST"]
_WEIGHTING_MATRIX = Literal[
    "weights_matrix_all",
    "weights_matrix_pd_gd",
    "weights_matrix_pd_md",
    "weights_matrix_gd_md",
    "gene_expression_correlation",
    "physical_distance",
    "morphological_distance",
]


def calculate_weight_matrix(
    adata: AnnData,
    adata_imputed: Union[AnnData, None] = None,
    pseudo_spots: bool = False,
    platform: _PLATFORM = "Visium",
) -> Optional[AnnData]:
    from sklearn.linear_model import LinearRegression
    import math

    if platform == "Visium":
        img_row = adata.obs["imagerow"]
        img_col = adata.obs["imagecol"]
        array_row = adata.obs["array_row"]
        array_col = adata.obs["array_col"]
        rate = 3
    elif platform == "Old_ST":
        img_row = adata.obs["imagerow"]
        img_col = adata.obs["imagecol"]
        array_row = adata.obs_names.map(lambda x: x.split("x")[1])
        array_col = adata.obs_names.map(lambda x: x.split("x")[0])
        rate = 1.5
    else:
        raise ValueError(
            f"""\
                {platform!r} does not support.
                """
        )

    reg_row = LinearRegression().fit(array_row.values.reshape(-1, 1), img_row)

    reg_col = LinearRegression().fit(array_col.values.reshape(-1, 1), img_col)

    if pseudo_spots and adata_imputed:
        pd = pairwise_distances(
            adata_imputed.obs[["imagecol", "imagerow"]],
            adata.obs[["imagecol", "imagerow"]],
            metric="euclidean",
        )
        unit = math.sqrt(reg_row.coef_**2 + reg_col.coef_**2)
        pd_norm = np.where(pd >= unit, 0, 1)

        md = 1 - pairwise_distances(
            adata_imputed.obsm["X_morphology"],
            adata.obsm["X_morphology"],
            metric="cosine",
        )
        md[md < 0] = 0

        adata_imputed.uns["physical_distance"] = pd_norm
        adata_imputed.uns["morphological_distance"] = md

        adata_imputed.uns["weights_matrix_all"] = (
            adata_imputed.uns["physical_distance"]
            * adata_imputed.uns["morphological_distance"]
        )

    else:
        pd = pairwise_distances(adata.obs[["imagecol", "imagerow"]], metric="euclidean")
        unit = math.sqrt(reg_row.coef_**2 + reg_col.coef_**2)
        pd_norm = np.where(pd >= rate * unit, 0, 1)

        md = 1 - pairwise_distances(adata.obsm["X_morphology"], metric="cosine")
        md[md < 0] = 0

        gd = 1 - pairwise_distances(adata.obsm["X_pca"], metric="correlation")
        adata.uns["gene_expression_correlation"] = gd
        adata.uns["physical_distance"] = pd_norm
        adata.uns["morphological_distance"] = md

        adata.uns["weights_matrix_all"] = (
            adata.uns["physical_distance"]
            * adata.uns["morphological_distance"]
            * adata.uns["gene_expression_correlation"]
        )
        adata.uns["weights_matrix_pd_gd"] = (
            adata.uns["physical_distance"] * adata.uns["gene_expression_correlation"]
        )
        adata.uns["weights_matrix_pd_md"] = (
            adata.uns["physical_distance"] * adata.uns["morphological_distance"]
        )
        adata.uns["weights_matrix_gd_md"] = (
            adata.uns["gene_expression_correlation"]
            * adata.uns["morphological_distance"]
        )


def impute_neighbour(
    adata: AnnData,
    count_embed: Union[np.ndarray, None] = None,
    weights: _WEIGHTING_MATRIX = "weights_matrix_all",
    copy: bool = False,
) -> Optional[AnnData]:
    coor = adata.obs[["imagecol", "imagerow"]]

    weights_matrix = adata.uns[weights]

    lag_coor = []

    weights_list = []

    with tqdm(
        total=len(adata),
        desc="Adjusting data",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for i in range(len(coor)):

            main_weights = weights_matrix[i]

            if weights == "physical_distance":
                current_neighbour = main_weights.argsort()[-6:]
            else:
                current_neighbour = main_weights.argsort()[-3:]

            surrounding_count = count_embed[current_neighbour]
            surrounding_weights = main_weights[current_neighbour]
            if surrounding_weights.sum() > 0:
                surrounding_weights_scaled = (
                    surrounding_weights / surrounding_weights.sum()
                )
                weights_list.append(surrounding_weights_scaled)

                surrounding_count_adjusted = np.multiply(
                    surrounding_weights_scaled.reshape(-1, 1), surrounding_count
                )
                surrounding_count_final = np.sum(surrounding_count_adjusted, axis=0)

            else:
                surrounding_count_final = np.zeros(count_embed.shape[1])
                weights_list.append(np.zeros(len(current_neighbour)))
            lag_coor.append(surrounding_count_final)
            pbar.update(1)

    imputed_data = np.array(lag_coor)
    key_added = "imputed_data"
    adata.obsm[key_added] = imputed_data

    adata.obsm["top_weights"] = np.array(weights_list)

    return adata if copy else None
