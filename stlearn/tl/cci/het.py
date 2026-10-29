"""Cell type heterogeneity: cell type counts per spot neighbourhood, and the
celltype-celltype interaction counts and permutation p-values behind run_cci.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from numba import njit, prange

from .neighbours import calc_distance, neighbourhood_matrix


def count(
    adata: AnnData,
    use_label: str | None = None,
    use_het: str = "cci_het",
    verbose: bool = True,
    distance: float | None = None,
    cell_prop_cutoff: float = 0.2,
) -> None:
    """Counts the cell types present in each spot's neighbourhood.

    A cell type is present in a spot when it makes up more than cell_prop_cutoff.
    The proportions are in adata.uns[use_label] or the label in adata.obs[use_label]
    for single cells.

    Parameters
    ----------
    adata: AnnData
        The data object including the cell types to count
    use_label:
        The cell type results to use in counting
    use_het:
        The storage place for result
    verbose: bool
        Whether to print where the result is stored.
    distance: float
        Distance to determine the neighbours (default is the nearest neighbour),
        distance=0 means within spot
    cell_prop_cutoff: float
        A cell type is present in a spot when its proportion is above this.

    Returns
    -------
    None
        The number of cell types in each spot's neighbourhood, the spot itself
        included, is stored in adata.obsm[use_het], one column.

    Raises
    ------
    KeyError
        If use_label is in neither adata.uns nor adata.obs.
    """

    present = label_proportions(adata, use_label).to_numpy() > cell_prop_cutoff
    # between spot
    if distance != 0:
        # The default neighbour distance if not given; distance=0 is within-spot.
        neighbourhoods = neighbourhood_matrix(adata, calc_distance(adata, distance))
        # The number of cell types present anywhere in each neighbourhood.
        adata.obsm[use_het] = ((neighbourhoods @ present.astype(np.int32)) > 0).sum(1)

    # within spot
    else:
        adata.obsm[use_het] = present.sum(axis=1)

    if verbose:
        print(
            "Counts for cluster (cell type) diversity stored into adata.obsm['"
            + use_het
            + "']",
        )


def label_keys(use_label: str) -> tuple[str, str]:
    """The obs and uns keys holding a label and its mixture proportions.

    Label transfer stores the dominant label in obs['predictions'] and the
    proportions in uns['label_transfer']; other labels use one key for both.
    """
    if use_label in ("label_transfer", "predictions"):
        return "predictions", "label_transfer"
    return use_label, use_label


def label_proportions(adata: AnnData, use_label: str) -> pd.DataFrame:
    """Spots by cell types: the proportions in adata.uns[use_label], as
    deconvolution and st.tl.cci.grid store them, or, without those, 1 for each
    spot's label in adata.obs[use_label] and 0 elsewhere, as for single cells."""
    obs_key, uns_key = label_keys(use_label)
    if uns_key in adata.uns:
        return adata.uns[uns_key].loc[adata.obs_names]
    if obs_key in adata.obs:
        return pd.get_dummies(adata.obs[obs_key], dtype=float)
    raise KeyError(
        f"No cell types {use_label!r} in adata.uns (proportions per spot) or "
        "adata.obs (a label per spot)."
    )


def get_data_for_counting(adata, use_label, mix_mode, all_set):
    """Spots by cell types (all_set): the cell type proportions in mixture mode,
    else 1 for each spot's label and 0 elsewhere."""
    # First determining how the edge counting needs to be performed #
    # Ensuring compatibility with current way of adding label_transfer to object
    obs_key, uns_key = label_keys(use_label)

    # Getting the cell type information; if not mixtures then populate
    # matrix with one's indicating pure spots.
    if mix_mode:
        cell_props = adata.uns[uns_key]
        # Exact names: "2" in "12" would read cluster 12's proportions for 2.
        cols = list(cell_props.columns.astype(str))
        col_order = [cols.index(cell_type) for cell_type in all_set]
        cell_data = cell_props.iloc[:, col_order].to_numpy(dtype=np.float64)
    else:
        cell_labels = adata.obs.loc[:, obs_key].values
        cell_data = np.zeros((len(cell_labels), len(all_set)), dtype=np.float64)
        for i, cell_type in enumerate(all_set):
            cell_data[:, i] = (
                (cell_labels == cell_type).astype(np.int32).astype(np.float64)
            )

    return cell_data


@njit(parallel=True)
def get_interaction_pvals(
    int_matrix,
    n_perms,
    cell_data,
    neighbourhood_indices,
    all_set,
    sig_bool,
    L_bool,
    R_bool,
    cell_prop_cutoff,
    seed,
):
    """Gets the p-values for the interaction counts."""

    # Counting how many times permutation of spots cell data creates interaction
    # counts greater than that observed, in order to calculate p-values.
    shape_ = (n_perms, int_matrix.shape[0], int_matrix.shape[1])
    # Storing the instances where the count is greater randomly for each perm.
    # Allows for embarassing parallelisation.
    greater_counts = np.zeros(shape_, dtype=np.int64)
    indices = np.zeros((cell_data.shape[0]), dtype=np.int64)
    for i in range(cell_data.shape[0]):
        indices[i] = i

    # If dealing with discrete data, no need to randomise columns indendently #
    discrete = np.all(np.logical_or(cell_data == 0, cell_data == 1))
    for i in prange(n_perms):
        # Each numba thread has its own random state, and a seed set outside prange
        # doesn't reach the threads drawing the permutations.
        np.random.seed(seed * n_perms + i)  # noqa: NPY002 (numba needs legacy API)
        # Permuting the cell data by swapping between spots for each column #
        if not discrete:
            perm_data = cell_data.copy()
            for j in range(cell_data.shape[1]):
                rand_indices = np.random.choice(  # noqa: NPY002
                    indices,
                    cell_data.shape[0],
                    False,
                )
                perm_data[:, j] = cell_data[rand_indices, j]
        else:
            rand_indices = np.random.choice(  # noqa: NPY002
                indices,
                cell_data.shape[0],
                False,
            )
            perm_data = cell_data[rand_indices, :]

        # Calculating interactions for permuted labels #
        perm_matrix = get_interaction_matrix(
            perm_data,
            neighbourhood_indices,
            all_set,
            sig_bool,
            L_bool,
            R_bool,
            cell_prop_cutoff,
        )
        for row in range(int_matrix.shape[0]):
            for col in range(int_matrix.shape[1]):
                greater_counts[i, row, col] = (
                    perm_matrix[row, col] >= int_matrix[row, col]
                )

    # Numba parallel sums axis 0 efficiently
    out = np.zeros((int_matrix.shape[0], int_matrix.shape[1]), dtype=np.float64)
    for i in range(n_perms):
        for row in range(int_matrix.shape[0]):
            for col in range(int_matrix.shape[1]):
                out[row, col] += greater_counts[i, row, col]
    # Counting the observed matrix as one of the permutations, so p is at least
    # 1 / (n_perms + 1) and never 0.
    int_pvals = (out + 1) / (n_perms + 1)
    return int_pvals


@njit
def get_interaction_matrix(
    cell_data,
    neighbourhood_indices,
    all_set,
    sig_bool,
    L_bool,
    R_bool,
    cell_prop_cutoff,
):
    """Gets the interaction matrix for a given cell data matrix."""

    n_spots = cell_data.shape[0]
    n_types = all_set.shape[0]
    int_matrix = np.zeros((n_types, n_types), dtype=np.int64)

    for t1 in range(n_types):
        for t2 in range(n_types):
            s = {np.int64(-1)}
            s.clear()

            for i in range(n_spots):
                if not sig_bool[i]:
                    continue
                if cell_data[i, t1] <= cell_prop_cutoff:
                    continue

                neighs = neighbourhood_indices[i][1]

                for k in range(len(neighs)):
                    n_idx = neighs[k]
                    if cell_data[n_idx, t2] > cell_prop_cutoff:
                        valid = False
                        if L_bool[i] and R_bool[n_idx]:
                            valid = True
                        if R_bool[i] and L_bool[n_idx]:
                            valid = True

                        if valid:
                            u = np.int64(i)
                            v = np.int64(n_idx)
                            if u > v:
                                tmp = u
                                u = v
                                v = tmp
                            s.add((u << 32) | v)

            int_matrix[t1, t2] = len(s)

    return int_matrix
