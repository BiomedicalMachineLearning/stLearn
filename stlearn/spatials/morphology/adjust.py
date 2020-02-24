from typing import Optional
import numpy as np
from anndata import AnnData
from ..._compat import Literal
import scipy.spatial as spatial


_SIMILARITY_MATRIX = Literal["cosine", "euclidean", "pearson"]


def adjust(
        adata: AnnData,
        use_count_data: str = "X_pca",
        use_image_data: str = "tile_feature_pca",
        key_added: str = "X_pca_morph_adj",
        radius: float = 50.0,
        rates: int = 1,
        copy: bool = False,
        similarity_matrix: _SIMILARITY_MATRIX = "cosine"

) -> Optional[AnnData]:
    coor = adata.obs[["imagecol", "imagerow"]]
    count_embed = adata.obsm[use_count_data]
    point_tree = spatial.cKDTree(coor)
    img_embed = adata.obsm[use_image_data]
    lag_coor = []

    for i in range(len(coor)):
        current_neightbor = point_tree.query_ball_point(coor.values[i], radius)  # Spatial weight
        current_neightbor.remove(i)

        main_count = count_embed[i].reshape(1, -1)
        main_img = img_embed[i].reshape(1, -1)
        surrounding_count = count_embed[current_neightbor]
        surrounding_img = img_embed[current_neightbor]

        similarity = []

        for i in surrounding_img:
            if similarity_matrix == "cosine":
                from sklearn.metrics.pairwise import cosine_similarity
                cosine = cosine_similarity(main_img, i.reshape(1, -1))[0][0]
                cosine = (abs(cosine)+cosine)/2
                similarity.append(cosine)
            elif similarity_matrix == "euclidean":
                pass
        similarity = np.array(similarity).reshape((-1, 1))
        surrounding_count_adjusted = np.multiply(surrounding_count, similarity)

        for i in range(0, rates):
            main_count = np.append(main_count, np.mean(surrounding_count_adjusted, axis=0).reshape(1, -1), axis=0)

        lag_coor.append(list(np.sum(main_count, axis=0)))

    adata.obsm[key_added] = np.array(lag_coor)

    return adata if copy else None

# TODO: add more distance matrix
