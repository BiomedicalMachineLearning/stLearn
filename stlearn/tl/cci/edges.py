"""Spot-to-spot edges of significant LR interactions, for plotting."""

import numpy as np
from anndata import AnnData

from .neighbours import stored_neighbours


def get_edges(
    adata: AnnData,
    l_bool: np.ndarray,
    r_bool: np.ndarray,
    sig_bool: np.ndarray,
) -> list[list[tuple[str, str]]]:
    """Gets the edges representing significant interactions, to draw.

    Parameters
    ----------
    adata : AnnData
        The data st.tl.cci.run stored its neighbours in, obsp["spot_neighbours"].
    l_bool : np.ndarray of bool, shape (n_spots,)
        Boolean array indicating spots where the ligand is expressed.
    r_bool : np.ndarray of bool, shape (n_spots,)
        Boolean array indicating spots where the receptor is expressed.
    sig_bool : np.ndarray of bool, shape (n_spots,)
        Boolean array indicating spots with significant ligand-receptor interactions.

    Returns
    -------
    [forward_edges, reverse_edges]: forward edges run from each significant spot
        expressing the ligand to its neighbours expressing the receptor, reverse
        edges from each significant spot expressing the receptor to its neighbours
        expressing the ligand. Each is a list of (barcode, barcode), with each
        pair of spots once, the way round it was first found.
    """
    graph = stored_neighbours(adata)
    names = adata.obs_names.to_numpy().astype(str)
    edge_lists = []
    for sources, targets in ((l_bool, r_bool), (r_bool, l_bool)):
        spots = np.flatnonzero(np.asarray(sources) & np.asarray(sig_bool))
        rows = graph[spots]
        starts = np.repeat(spots, np.diff(rows.indptr))
        ends = rows.indices[np.asarray(targets)[rows.indices]]
        starts = starts[np.asarray(targets)[rows.indices]]
        # Each pair of spots once, in the order found: direction doesn't matter.
        pairs = np.minimum(starts, ends).astype(np.int64) * graph.shape[0]
        pairs += np.maximum(starts, ends)
        _, first = np.unique(pairs, return_index=True)
        first.sort()
        edge_lists.append(
            list(zip(names[starts[first]], names[ends[first]], strict=True))
        )
    return edge_lists
