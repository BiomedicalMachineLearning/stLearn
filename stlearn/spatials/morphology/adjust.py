from typing import Optional
import numpy as np
from anndata import AnnData
from ..._compat import Literal
import scipy.spatial as spatial
from scipy import stats
# Test progress bar
from tqdm import tqdm

_SIMILARITY_MATRIX = Literal["cosine", "euclidean", "pearson"]


def adjust(
        adata: AnnData,
        use_data: str = "X_pca",
        #use_image_data: str = "X_morphology",
        #key_added: str = "X_pca_morph_adj",
        radius: float = 50.0,
        rates: int = 1,
        method = "mean",
        copy: bool = False,
        similarity_matrix: _SIMILARITY_MATRIX = "cosine"

) -> Optional[AnnData]:

    if "X_morphology" not in adata.obsm:
        raise ValueError("Please run the function stlearn.pp.extract_feature")
    coor = adata.obs[["imagecol", "imagerow"]]
    count_embed = adata.obsm[use_data]
    point_tree = spatial.cKDTree(coor)
    img_embed = adata.obsm["X_morphology"]
    lag_coor = []
    with tqdm(total=len(adata), desc="Adjusting data", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for i in range(len(coor)):
            current_neightbor = point_tree.query_ball_point(coor.values[i], radius)  # Spatial weight
            current_neightbor.remove(i)

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
                    cosine = (abs(cosine)+cosine)/2
                    similarity.append(cosine)

                elif similarity_matrix == "euclidean":
                    from sklearn.metrics.pairwise import euclidean_distances
                    eculidean = euclidean_distances(main_img, i)[0][0]
                    eculidean_similarity = 1 / (1 + eculidean)
                    similarity.append(eculidean_similarity)

                elif similarity_matrix == "pearson":
                    pass

            similarity = np.array(similarity).reshape((-1, 1))
            surrounding_count_adjusted = np.multiply(surrounding_count, similarity)

            for i in range(0, rates):
                if method == "median":
                    main_count = np.append(main_count, np.median(surrounding_count_adjusted, axis=0).reshape(1, -1), axis=0)
                elif method == "mean":
                    main_count = np.append(main_count, np.mean(surrounding_count_adjusted, axis=0).reshape(1, -1), axis=0)
            lag_coor.append(list(np.sum(main_count, axis=0)))
            pbar.update(1)
    key_added = use_data + "_morphology"
    adata.obsm[key_added] = np.array(lag_coor)

    print("The data adjusted by morphology is added to adata.obsm['" + key_added + "']")

    return adata if copy else None

# TODO: add more distance matrix
