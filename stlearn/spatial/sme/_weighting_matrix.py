"""Neighbour weights for SME normalisation, imputation and pseudo-spots.

Each spot is adjusted by its top few neighbours, ranked by a weight that
multiplies some of:

- physical distance (pd): 1 within a radius of the spot, else 0;
- morphological similarity (md): cosine similarity of `obsm["X_morphology"]`,
  clipped to [0, 1];
- gene expression correlation (gd): Pearson correlation of `obsm["X_pca"]`.
"""

import math
from typing import Literal

import numpy as np
from anndata import AnnData
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression  # type: ignore
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

# The similarities each weighting multiplies.
_FACTORS: dict[str, tuple[str, ...]] = {
    "weights_matrix_all": ("pd", "md", "gd"),
    "weights_matrix_pd_gd": ("pd", "gd"),
    "weights_matrix_pd_md": ("pd", "md"),
    "weights_matrix_gd_md": ("md", "gd"),
    "gene_expression_correlation": ("gd",),
    "physical_distance": ("pd",),
    "morphological_distance": ("md",),
}
# Pseudo-spots have no expression, so their weights leave out gd; their
# "weights_matrix_all" is physical distance x morphology.
_PSEUDO_SPOT_FACTORS: dict[str, tuple[str, ...]] = {
    "weights_matrix_all": ("pd", "md"),
    "weights_matrix_pd_md": ("pd", "md"),
    "physical_distance": ("pd",),
    "morphological_distance": ("md",),
}

# Spots per block when every spot is a candidate neighbour: a block's weights
# against all spots take this many bytes, and its temporaries a few times more.
_BLOCK_BYTES = 64 * 1024**2
# Spots per chunk when candidates are the spots within the radius.
_CHUNK_SPOTS = 4096


def _unit(reg_col: LinearRegression, reg_row: LinearRegression) -> float:
    """Image distance of one step along both array axes."""
    return math.sqrt(reg_row.coef_[0] ** 2 + reg_col.coef_[0] ** 2)


def neighbour_weights(
    adata: AnnData,
    weights: _WEIGHTING_MATRIX = "weights_matrix_all",
    platform: _PLATFORM = "Visium",
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Each spot's top neighbours among the spots of `adata`, and their weights.

    The neighbours are the 3 spots (6 for "physical_distance") with the largest
    weight, the spot itself included.

    Returns
    -------
    neighbours
        Spots x spots sparse matrix: row i holds spot i's neighbours, with
        their weights scaled to sum to 1. A row is empty when the weights
        don't sum to more than 0.
    top_weights
        Spots x 3 (or 6): the scaled weights of each spot's neighbours,
        smallest first, as stored in `obsm["top_weights"]`.
    """
    if weights not in _FACTORS:
        raise ValueError(f"weights must be one of {list(_FACTORS)}, got {weights!r}.")
    factors = _FACTORS[weights]
    radius = None
    if "pd" in factors:
        reg_col, reg_row, rate = row_col_by_platform(adata, platform)
        radius = rate * _unit(reg_col, reg_row)
    return _top_neighbours(
        adata,
        adata,
        factors,
        radius=radius,
        k=6 if weights == "physical_distance" else 3,
        same=True,
    )


def pseudo_spot_weights(
    adata: AnnData,
    pseudo_spots: AnnData,
    weights: _WEIGHTING_MATRIX = "weights_matrix_all",
    platform: _PLATFORM = "Visium",
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Each pseudo-spot's top neighbours among the spots of `adata`.

    As `neighbour_weights`, with pseudo-spots x spots `neighbours`. Physical
    distance counts spots within one spot step; there is no expression
    correlation, as pseudo-spots have no expression yet.
    """
    if weights not in _PSEUDO_SPOT_FACTORS:
        raise ValueError(
            f"Pseudo-spots can be weighted by {list(_PSEUDO_SPOT_FACTORS)}, "
            f"not {weights!r}: they have no expression to correlate."
        )
    factors = _PSEUDO_SPOT_FACTORS[weights]
    radius = None
    if "pd" in factors:
        reg_col, reg_row, _ = row_col_by_platform(adata, platform)
        radius = _unit(reg_col, reg_row)
    return _top_neighbours(
        pseudo_spots,
        adata,
        factors,
        radius=radius,
        k=6 if weights == "physical_distance" else 3,
        same=False,
    )


def _coordinates(adata: AnnData) -> np.ndarray:
    return adata.obs[["imagecol", "imagerow"]].to_numpy(dtype=np.float64)


def _unit_rows(matrix: np.ndarray, centre: bool) -> np.ndarray:
    """Rows scaled to length 1 (after centring them, for correlation), in
    float64. Rows of zeros stay zeros, as sklearn's cosine similarity has them."""
    rows = np.asarray(matrix, dtype=np.float64)
    if centre:
        rows = rows - rows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return rows / norms


def _md(cosine: np.ndarray) -> np.ndarray:
    """Morphological similarity from cosine similarity, as 1 - sklearn's
    cosine distance (clipped to [0, 2]), with negatives set to 0."""
    md = 1 - np.clip(1 - cosine, 0, 2)
    md[md < 0] = 0
    return md


def _gd(correlation: np.ndarray) -> np.ndarray:
    """Expression similarity from correlation, as 1 - scipy's correlation
    distance."""
    return 1 - (1 - correlation)


def _top_neighbours(
    query: AnnData,
    reference: AnnData,
    factors: tuple[str, ...],
    radius: float | None,
    k: int,
    same: bool,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """For each spot of `query`, the k spots of `reference` with the largest
    weight, as SME has always picked them: the k largest values of the spot's
    full row of weights, where spots outside the radius weigh 0.

    Their weights are scaled by their sum when it is above 0. `same` is True
    when query and reference are the same spots: a spot's similarity to
    itself is then exactly 1.
    """
    n_query, n_reference = query.n_obs, reference.n_obs
    k = min(k, n_reference)
    features = {}
    for name, key, centre in (("md", "X_morphology", False), ("gd", "X_pca", True)):
        if name in factors:
            query_units = _unit_rows(query.obsm[key], centre=centre)
            reference_units = (
                query_units if same else _unit_rows(reference.obsm[key], centre=centre)
            )
            features[name] = (query_units, reference_units)

    rows, columns, values = [], [], []
    top_weights = np.zeros((n_query, k))
    if radius is not None:
        chunks = _within_radius(query, reference, radius, features, same)
    else:
        chunks = _all_pairs(n_query, n_reference, features, same)
    with tqdm(
        total=n_query,
        desc="Adjusting data",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for start, stop, chunk in chunks:
            chunk_rows, chunk_columns, chunk_values, chunk_top = chunk(k)
            rows.append(chunk_rows + start)
            columns.append(chunk_columns)
            values.append(chunk_values)
            top_weights[start:stop] = chunk_top
            pbar.update(stop - start)

    neighbours = sparse.csr_matrix(
        (
            np.concatenate([np.zeros(0), *values]),
            (
                np.concatenate([np.zeros(0, dtype=np.int64), *rows]),
                np.concatenate([np.zeros(0, dtype=np.int64), *columns]),
            ),
        ),
        shape=(n_query, n_reference),
    )
    return neighbours, top_weights


def _within_radius(query, reference, radius, features, same):
    """Chunks of query spots, with candidates the reference spots closer than
    `radius`; every other spot weighs 0."""
    reference_tree = cKDTree(_coordinates(reference))
    coordinates = _coordinates(query)
    n_reference = reference.n_obs
    for start in range(0, query.n_obs, _CHUNK_SPOTS):
        stop = min(start + _CHUNK_SPOTS, query.n_obs)

        def chunk(k, start=start, stop=stop):
            pairs = cKDTree(coordinates[start:stop]).sparse_distance_matrix(
                reference_tree, radius, output_type="ndarray"
            )
            pairs = pairs[pairs["v"] < radius]  # pd is 1 strictly within
            row, column = pairs["i"].astype(np.int64), pairs["j"].astype(np.int64)
            weight = np.ones(len(pairs))
            for name, (query_units, reference_units) in features.items():
                similarity = np.einsum(
                    "ij,ij->i",
                    query_units[start:stop][row],
                    reference_units[column],
                )
                if same:
                    similarity[row + start == column] = 1.0
                weight = weight * (_md(similarity) if name == "md" else _gd(similarity))
            return _pick(row, column, weight, pairs["v"], stop - start, n_reference, k)

        yield start, stop, chunk


def _all_pairs(n_query, n_reference, features, same):
    """Blocks of query spots, with every reference spot a candidate."""
    block = max(1, _BLOCK_BYTES // (8 * max(n_reference, 1)))
    for start in range(0, n_query, block):
        stop = min(start + block, n_query)

        def chunk(k, start=start, stop=stop):
            weight = np.ones((stop - start, n_reference))
            for name, (query_units, reference_units) in features.items():
                similarity = query_units[start:stop] @ reference_units.T
                if same:
                    similarity[np.arange(stop - start), np.arange(start, stop)] = 1.0
                weight *= _md(similarity) if name == "md" else _gd(similarity)
            return _pick_dense(weight, k)

        yield start, stop, chunk


def _pick(row, column, weight, distance, n_rows, n_reference, k):
    """The k largest weights of each row, from candidate pairs.

    Rows' other entries weigh 0, so after a row's positive candidates come
    zeros - candidates weighing 0 and every non-candidate - and only then
    negative candidates. Ties go to the nearer spot, then the lower index.
    """
    order = np.lexsort((column, distance, -weight, row))
    row, column, weight = row[order], column[order], weight[order]

    n_candidates = np.bincount(row, minlength=n_rows)
    n_positive = np.bincount(row[weight > 0], minlength=n_rows)
    n_zero = np.bincount(row[weight == 0], minlength=n_rows)
    zeros = n_zero + (n_reference - n_candidates)
    rank = _rank_in_row(row, n_rows)

    positive = (weight > 0) & (rank < k)
    negatives_needed = np.maximum(0, k - np.minimum(n_positive, k) - zeros)
    negative_rank = rank - n_positive[row] - n_zero[row]
    negative = (weight < 0) & (negative_rank < negatives_needed[row])
    picked = positive | negative

    row, column, weight = row[picked], column[picked], weight[picked]
    total = np.bincount(row, weights=weight, minlength=n_rows)
    kept = total[row] > 0
    row, column = row[kept], column[kept]
    scaled = weight[kept] / total[row]

    # Each row's k scaled weights, with zeros for the zeros picked, smallest first.
    top = np.zeros((n_rows, k))
    top[row, _rank_in_row(row, n_rows)] = scaled
    top.sort(axis=1)
    return row, column, scaled, top


def _rank_in_row(row: np.ndarray, n_rows: int) -> np.ndarray:
    """Position of each entry within its row, for entries sorted by row."""
    counts = np.bincount(row, minlength=n_rows)
    first = np.cumsum(counts) - counts
    return np.arange(len(row)) - first[row]


def _pick_dense(weight, k):
    """The k largest weights of each row of a block of full rows."""
    n_rows = weight.shape[0]
    top_columns = np.argpartition(-weight, k - 1, axis=1)[:, :k]
    top_values = np.take_along_axis(weight, top_columns, axis=1)
    total = top_values.sum(axis=1)
    kept_rows = total > 0
    scaled = np.zeros_like(top_values)
    scaled[kept_rows] = top_values[kept_rows] / total[kept_rows, None]

    nonzero = scaled != 0
    row = np.broadcast_to(np.arange(n_rows)[:, None], scaled.shape)[nonzero]
    column = top_columns[nonzero]
    top = np.sort(scaled, axis=1)
    return row, column, scaled[nonzero], top


def row_col_by_platform(
    adata, platform
) -> tuple[LinearRegression, LinearRegression, float]:
    rate: float
    if platform == "Visium":
        img_row = adata.obs["imagerow"]
        img_col = adata.obs["imagecol"]
        array_row = adata.obs["array_row"]
        array_col = adata.obs["array_col"]
        rate = 3
    elif platform == "Old_ST":
        img_row = adata.obs["imagerow"]
        img_col = adata.obs["imagecol"]
        array_row = adata.obs_names.map(lambda x: float(x.split("x")[1]))
        array_col = adata.obs_names.map(lambda x: float(x.split("x")[0]))
        rate = 1.5
    else:
        raise ValueError(
            f"""\
                {platform!r} does not support.
                """
        )
    # One estimator per axis: fit() returns the estimator itself, so fitting a
    # shared one twice leaves both names pointing at the column fit.
    reg_row: LinearRegression = LinearRegression().fit(  # type: ignore
        array_row.values.reshape(-1, 1), img_row
    )
    reg_col: LinearRegression = LinearRegression().fit(  # type: ignore
        array_col.values.reshape(-1, 1), img_col
    )
    return reg_col, reg_row, rate
