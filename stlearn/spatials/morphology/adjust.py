from typing import Literal

import numpy as np
import scipy.spatial as spatial
from anndata import AnnData
from tqdm import tqdm

_SIMILARITY_MATRIX = Literal["cosine", "euclidean", "pearson", "spearman"]
_METHOD = Literal["mean", "median", "sum"]

def adjust(
    adata: AnnData,
    use_data: str = "X_pca",
    radius: float = 50.0,
    rates: int = 1,
    method: _SIMILARITY_MATRIX = "mean",
    similarity_matrix: _SIMILARITY_MATRIX = "cosine",
    copy: bool = False,
) -> AnnData | None:
    """\
    SME normalisation: Using spot location information and tissue morphological
    features to correct spot gene expression

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    use_data : str, default "X_pca"
        Input date to be adjusted by morphological features.
        choose one from ["raw", "X_pca", "X_umap"]
    radius: float, default 50.0
        Radius to select neighbour spots.
    rates: int, default 1
        Number of times to add the aggregated neighbor contribution.
        Higher values increase the strength of morphological adjustment.
    method: {'mean', 'median', 'sum'}, default 'mean'
        Method for aggregating neighbor contributions.
    similarity_matrix : {'cosine', 'euclidean', 'pearson', 'spearman'}, default 'cosine'
        Method to calculate morphological similarity between spots.
    copy : bool, default False
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **[use_data]_morphology** : `adata.obsm` field
        Add SME normalised gene expression matrix
    """
    adata = adata.copy() if copy else adata

    if "X_morphology" not in adata.obsm:
        raise ValueError("Please run the function stlearn.pp.extract_feature")
    coor = adata.obs[["imagecol", "imagerow"]]
    if use_data == "raw":
        count_embed = adata.X
    else:
        count_embed = adata.obsm[use_data]
    point_tree = spatial.cKDTree(coor)
    img_embed = adata.obsm["X_morphology"]
    lag_coor = []
    with tqdm(
        total=len(adata),
        desc="Adjusting data",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for i in range(len(coor)):
            current_neightbor = point_tree.query_ball_point(
                coor.values[i], radius
            )  # Spatial weight
            current_neightbor.remove(i)
            if len(current_neightbor) > 0:
                main_count = count_embed[i].reshape(1, -1)
                main_img = img_embed[i].reshape(1, -1)
                surrounding_count = count_embed[current_neightbor]
                surrounding_img = img_embed[current_neightbor]

                similarity = []

                for i in surrounding_img:
                    i = i.reshape(1, -1)  # reshape feature to (1, n_feature)

                    if similarity_matrix == "cosine":
                        from sklearn.metrics.pairwise import cosine_similarity

                        cosine = cosine_similarity(main_img, i)[0][0]
                        cosine = (abs(cosine) + cosine) / 2
                        similarity.append(cosine)
                    elif similarity_matrix == "euclidean":
                        from sklearn.metrics.pairwise import euclidean_distances

                        eculidean = euclidean_distances(main_img, i)[0][0]
                        eculidean = 1 / (1 + eculidean)
                        similarity.append(eculidean)
                    elif similarity_matrix == "pearson":
                        from scipy.stats import pearsonr

                        pearson_corr = abs(
                            pearsonr(main_img.reshape(-1), i.reshape(-1))[0]
                        )
                        similarity.append(pearson_corr)
                    elif similarity_matrix == "spearman":
                        from scipy.stats import spearmanr

                        spearmanr_corr = abs(
                            spearmanr(main_img.reshape(-1), i.reshape(-1))[0]
                        )
                        similarity.append(spearmanr_corr)

                similarity = np.array(similarity).reshape((-1, 1))
                surrounding_count_adjusted = np.multiply(surrounding_count, similarity)

                for i in range(0, rates):
                    if method == "median":
                        main_count = np.append(
                            main_count,
                            np.median(surrounding_count_adjusted, axis=0).reshape(
                                1, -1
                            ),
                            axis=0,
                        )
                    elif method == "mean":
                        main_count = np.append(
                            main_count,
                            np.mean(surrounding_count_adjusted, axis=0).reshape(1, -1),
                            axis=0,
                        )
                    elif method == "sum":
                        main_count = np.append(
                            main_count,
                            np.sum(surrounding_count_adjusted, axis=0).reshape(1, -1),
                            axis=0,
                        )
                    else:
                        raise ValueError("Only 'median' and 'mean' are aceptable")
                lag_coor.append(list(np.sum(main_count, axis=0)))
                pbar.update(1)
            else:
                lag_coor.append(list(np.sum(main_count, axis=0)))
                pbar.update(1)
    key_added = use_data + "_morphology"
    adata.obsm[key_added] = np.array(lag_coor)

    print("The data adjusted by morphology is added to adata.obsm['" + key_added + "']")

    return adata if copy else None
