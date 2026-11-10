"""The layout both acceptance samples have: 8 µm Visium HD bins on a square grid,
with the barcodes, positions and scale factors Space Ranger gives them."""

import numpy as np
from anndata import AnnData

# 8 µm bins imaged at 0.2738 µm per pixel, hires image at 0.08 of full size.
SCALEFACTORS = {
    "bin_size_um": 8.0,
    "microns_per_pixel": 0.2738,
    "spot_diameter_fullres": 8.0 / 0.2738,
    "tissue_hires_scalef": 0.08,
    "tissue_lowres_scalef": 0.024,
}


def bin_grid(n_bins: int) -> tuple[int, np.ndarray, np.ndarray]:
    """The side of the square grid of about `n_bins` bins, and each bin's row and
    column, row by row."""
    n_side = round(np.sqrt(n_bins))
    row, col = np.divmod(np.arange(n_side * n_side), n_side)
    return n_side, row, col


def barcodes(row: np.ndarray, col: np.ndarray) -> list[str]:
    return [f"s_008um_{r:05d}_{c:05d}-1" for r, c in zip(row, col, strict=True)]


def place_bins(adata: AnnData, row: np.ndarray, col: np.ndarray) -> None:
    """The bins' positions, in full-resolution pixels (obsm["spatial"]) and hires
    pixels (obs imagecol and imagerow), and the scale factors in uns["spatial"]."""
    spot = SCALEFACTORS["spot_diameter_fullres"]
    hires = SCALEFACTORS["tissue_hires_scalef"]
    adata.obsm["spatial"] = np.column_stack([col, row]) * spot + 1000
    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * hires
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * hires
    adata.uns["spatial"] = {
        "synthetic_hd": {
            "images": {},
            "scalefactors": SCALEFACTORS,
            "use_quality": "hires",
        }
    }
