"""Gridding single-cell spatial data into squares, upstream of the CCI pipeline.

st.tl.cci.create_grid sizes the squares in microns; st.tl.cci.grid takes the
number of rows and columns.
"""

import numpy as np
import pandas as pd
import scipy.sparse
from anndata import AnnData
from scipy.sparse import spmatrix

from ...utils import (
    _image_coords,
    _image_scalef,
    _microns_per_image_pixel,
    get_shape,
)


def create_grid(
    adata: AnnData,
    bin_um: float = 52.0,
    use_label: str | None = None,
    microns_per_pixel: float | None = None,
    n_cpus: int | None = None,
    verbose: bool = True,
) -> AnnData:
    """Grids single-cell data into squares about `bin_um` microns across, with
        st.tl.cci.grid.

    st.tl.cci.grid splits the box around the cells into a set number of rows and
    columns, so the size of its squares depends on the size of the tissue: 125 by
    125 on a section taller than it is wide gives tall rectangles. This picks the
    rows and columns from the tissue's extent in microns instead, so the squares
    are square and the same size on any sample or crop.

    Parameters
    ----------
    adata: AnnData
        The data object, with obs["imagecol"], obs["imagerow"] and uns["spatial"],
        as st.io.attach_spatial_sdata, st.io.attach_spatial and st.io.read_xenium
        set them.
    bin_um: float
        Side of a grid square in microns. The default, 52, is 125 squares across
        a 6.5 mm Visium capture area, close to the 55 µm of a Visium spot. Each
        side of the tissue is rounded to a whole number of squares, so with n
        squares along a side they are within 1 / (2n) of bin_um;
        uns["grid_bin_um"] has the sizes used.
    use_label: str
        The cell type labels in adata.obs to join together & save as deconvolution
        data, as for st.tl.cci.grid.
    microns_per_pixel: float
        Microns per full-resolution pixel. If None, uses
        uns["spatial"][library_id]["scalefactors"]["microns_per_pixel"], which
        the st.io readers set and Space Ranger writes for Visium HD.
    n_cpus: int
        Number of threads to use or if None use numba.config.NUMBA_NUM_THREADS
    verbose: bool
        Whether to print the grid's size.
    Returns
    -------
    grid_data: AnnData
        As st.tl.cci.grid returns it, with uns["grid_bin_um"], the squares' width
        and height in microns.
    Raises
    ------
    ValueError
        If bin_um is not positive, or there is no microns_per_pixel.
    """
    if not bin_um > 0:
        raise ValueError(f"bin_um must be positive, not {bin_um}.")

    um_per_image_px = _microns_per_image_pixel(adata, microns_per_pixel)
    n_col, n_row = get_shape(adata, bin_um, microns_per_pixel)

    grid_data = grid(
        adata,
        n_row=int(n_row),
        n_col=int(n_col),
        use_label=use_label,
        verbose=False,
    )
    grid_data.uns["grid_bin_um"] = um_per_image_px * np.array(
        [
            np.diff(grid_data.uns["grid_xedges"])[0],
            np.diff(grid_data.uns["grid_yedges"])[0],
        ]
    )
    if verbose:
        width, height = grid_data.uns["grid_bin_um"]
        print(
            f"Gridding into {n_row} rows by {n_col} columns of {width:.1f} by "
            f"{height:.1f} µm squares; {grid_data.n_obs:,} have cells."
        )
    return grid_data


def grid(
    adata: AnnData,
    n_row: int = 10,
    n_col: int = 10,
    use_label: str | None = None,
    verbose: bool = True,
) -> AnnData:
    """Creates a new anndata representing a gridded version of the data; can be
        used upstream of CCI pipeline. NOTE: intended use is for single cell
        spatial data, not Visium or other lower resolution tech.

    The squares split the box around the cells into n_row by n_col, so their size
    depends on the tissue's; st.tl.cci.create_grid sizes them in microns.

    Parameters
    ----------
    adata: AnnData
        The data object.
    n_row: int
        The number of rows in the grid.
    n_col: int
        The number of columns in the grid.
    use_label: str
        The cell type labels in adata.obs to join together & save as deconvolution data.
    Returns
    -------
    grid_data: AnnData
        Equivalent expression data to adata, except values have been summed by
        cells that fall within defined bins.
    """
    if verbose:
        print("Gridding...")

    # Retrieving the coordinates of each grid #
    n_squares = n_row * n_col
    xs, ys = _image_coords(adata)

    grid_counts, xedges, yedges = np.histogram2d(xs, ys, bins=[n_col, n_row])
    grid_counts, xedges, yedges = (
        grid_counts.astype(int),
        xedges.astype(float),
        yedges.astype(float),
    )

    grid_expr = np.zeros((n_squares, adata.shape[1]))
    grid_coords = np.zeros((n_squares, 2))
    grid_cell_counts = np.zeros(n_squares, dtype=np.int64)
    # If use_label is specified, then it will generate deconvolution information
    cell_labels, cell_set, cell_info = None, None, None
    if use_label is not None:
        cell_labels = adata.obs[use_label].to_numpy().astype(str)
        cell_set = np.unique(cell_labels).astype(str)
        cell_info = np.zeros((n_squares, len(cell_set)), dtype=np.float64)

    # Assigning cells to grid squares #
    bin_cells(
        grid_coords,
        xedges,
        yedges,
        n_row,
        n_col,
        xs,
        ys,
        grid_cell_counts,
        grid_expr,
        adata.X,
        use_label is not None,
        cell_labels,
        cell_info,
        cell_set,
    )

    # Creating gridded anndata #
    grid_expr = pd.DataFrame(
        grid_expr,
        index=[f"grid_{i}" for i in range(n_squares)],
        columns=adata.var_names.values.astype(str),
    )
    grid_data = AnnData(grid_expr)
    grid_data.obs["imagecol"] = grid_coords[:, 0]
    grid_data.obs["imagerow"] = grid_coords[:, 1]
    grid_data.obs["n_cells"] = grid_cell_counts
    grid_data.uns["spatial"] = adata.uns["spatial"]

    # grid_coords are in image pixels, imagecol/imagerow's units; obsm["spatial"] is
    # in full-resolution units, which the image's scale factor maps onto the image.
    grid_data.obsm["spatial"] = grid_coords / _image_scalef(adata)

    if use_label is not None and cell_info is not None and cell_set is not None:
        grid_data.uns[use_label] = pd.DataFrame(
            cell_info,
            index=grid_data.obs_names.values.astype(str),
            columns=cell_set,
        )
        max_indices = np.apply_along_axis(np.argmax, 1, cell_info)
        grid_data.obs[use_label] = [cell_set[index] for index in max_indices]
        grid_data.obs[use_label] = grid_data.obs[use_label].astype("category")
        grid_data.obs[use_label] = grid_data.obs[use_label].cat.set_categories(
            adata.obs[use_label].cat.categories,
        )
        if f"{use_label}_colors" in adata.uns:
            grid_data.uns[f"{use_label}_colors"] = adata.uns[f"{use_label}_colors"]

    # Subsetting to only gridded spots that contain cells #
    grid_data = grid_data[grid_data.obs["n_cells"] > 0, :].copy()
    if use_label is not None:
        grid_data.uns[use_label] = grid_data.uns[use_label].loc[grid_data.obs_names, :]

    grid_data.uns["grid_counts"] = grid_counts
    grid_data.uns["grid_xedges"] = xedges
    grid_data.uns["grid_yedges"] = yedges

    return grid_data


def bin_cells(
    grid_coords: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
    n_row: int,
    n_col: int,
    xs: np.ndarray,
    ys: np.ndarray,
    grid_cell_counts: np.ndarray,
    grid_expr: np.ndarray,
    cell_expr: np.ndarray | spmatrix,
    use_label_bool: bool,
    cell_labels: np.ndarray,
    cell_info: np.ndarray,
    cell_set: np.ndarray,
):
    """Grids the gene expression information, filling the output arrays in place.

    Square n = i * n_row + j is column i of xedges and row j of yedges. Cells are
    binned as numpy.histogram2d bins them: a square holds its lower edges, and the
    last square along each axis its upper edge too, so every cell is in one square.
    cell_set must be sorted, as numpy.unique returns it.
    """
    n_cells, n_squares = len(xs), n_col * n_row
    squares = np.digitize(xs, xedges[1:-1]) * n_row + np.digitize(ys, yedges[1:-1])

    grid_coords[:, 0] = np.repeat((xedges[:-1] + xedges[1:]) / 2, n_row)
    grid_coords[:, 1] = np.tile((yedges[:-1] + yedges[1:]) / 2, n_col)
    grid_cell_counts[:] = np.bincount(squares, minlength=n_squares)

    # Squares by cells: 1 where the cell is in the square.
    membership = scipy.sparse.csr_matrix(
        (np.ones(n_cells), (squares, np.arange(n_cells))), shape=(n_squares, n_cells)
    )
    summed = membership @ cell_expr
    grid_expr[:] = summed.toarray() if scipy.sparse.issparse(summed) else summed

    if use_label_bool:
        type_counts = membership @ np.equal.outer(cell_labels, cell_set)
        cell_info[:] = type_counts / np.maximum(grid_cell_counts, 1)[:, np.newaxis]
