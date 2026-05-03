import numpy as np
import scipy.sparse as sp
import scipy.spatial as spatial
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm

from stlearn.types import _METHOD, _SIMILARITY_MATRIX


def _make_similarity_fn(name):
    if name == "cosine":

        def fn(ref, neighbours):
            sims = cosine_similarity(ref, neighbours)[0]
            return np.clip(sims, 0, None)  # max(sim, 0)

        return fn
    if name == "euclidean":

        def fn(ref, neighbours):
            d = euclidean_distances(ref, neighbours)[0]
            return 1 / (1 + d)

        return fn
    if name == "pearson":

        def fn(ref, neighbours):
            ref_flat = ref.reshape(-1)
            return np.array([abs(pearsonr(ref_flat, n)[0]) for n in neighbours])

        return fn
    if name == "spearman":

        def fn(ref, neighbours):
            ref_flat = ref.reshape(-1)
            return np.array([abs(spearmanr(ref_flat, n)[0]) for n in neighbours])

        return fn
    raise ValueError(f"Unknown similarity_matrix: {name!r}")


def _row_as_dense(matrix, idx):
    """Extract row(s) as a dense 2D ndarray, regardless of sparse/dense input."""
    row = matrix[idx]
    if sp.issparse(row):
        row = row.toarray()
    return np.atleast_2d(row)


def adjust(
    adata: AnnData,
    use_data: str = "X_pca",
    radius: float = 50.0,
    rates: int = 1,
    method: _METHOD = "mean",
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

    if method == "mean":
        reducer = np.mean
    elif method == "median":
        reducer = np.median
    elif method == "sum":
        reducer = np.sum
    else:
        raise ValueError("Only 'median', 'sum', and 'mean' are acceptable")

    similarity_fn = _make_similarity_fn(similarity_matrix)

    coords = adata.obs[["imagecol", "imagerow"]]
    if use_data == "raw":
        count_embed = adata.X
    else:
        count_embed = adata.obsm[use_data]
    is_sparse = sp.issparse(count_embed)
    # Convert to CSR once if sparse — fast row indexing
    if is_sparse and not sp.isspmatrix_csr(count_embed):
        count_embed = count_embed.tocsr()

    point_tree = spatial.cKDTree(coords)
    img_embed = adata.obsm["X_morphology"]

    n_points = len(coords)
    n_features = count_embed.shape[1]
    out_dtype = count_embed.dtype if not is_sparse else np.float64
    lag_coords = np.empty((n_points, n_features), dtype=out_dtype)

    for i in tqdm(
        range(n_points),
        desc="Adjusting data",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ):
        neighbours = point_tree.query_ball_point(coords.values[i], radius)
        neighbours.remove(i)

        main_count = _row_as_dense(count_embed, i)

        if not neighbours:
            lag_coords[i] = main_count.sum(axis=0)
            continue

        main_img = img_embed[i].reshape(1, -1)
        surrounding_count = _row_as_dense(count_embed, neighbours)
        surrounding_img = img_embed[neighbours]

        similarity = similarity_fn(main_img, surrounding_img).reshape(-1, 1)
        surrounding_count_adjusted = surrounding_count * similarity

        # Aggregate neighbour contribution and replicate `rates` times
        aggregated = reducer(surrounding_count_adjusted, axis=0).reshape(1, -1)
        stacked = np.concatenate(
            [main_count, np.tile(aggregated, (rates, 1))],
            axis=0,
        )
        lag_coords[i] = stacked.sum(axis=0)

    key_added = use_data + "_morphology"
    adata.obsm[key_added] = lag_coords

    print("The data adjusted by morphology is added to adata.obsm['" + key_added + "']")

    return adata if copy else None
