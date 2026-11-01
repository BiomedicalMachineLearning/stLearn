import math
import warnings

import anndata
import numpy as np
from skimage.transform import resize

from ..utils import _image_scalef, as_uint8_image


def _sample_image(adata):
    """The library id, its `uns["spatial"]` entry, the image of `use_quality` and its
    scale factor (image pixels per `obsm["spatial"]` unit)."""
    library_id = next(iter(adata.uns["spatial"].keys()))
    spatial = adata.uns["spatial"][library_id]
    quality = spatial["use_quality"]
    scale = _image_scalef(adata)
    return library_id, spatial, spatial["images"][quality], scale


def _as_float_rgb(image):
    """An image as (y, x, 3) float32 RGB in [0, 1], for resizing."""
    image = as_uint8_image(image)  # Any dtype stLearn reads images in, to 0-255.
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = image[..., :3]  # drop alpha
    return image.astype(np.float32) / 255


def concatenate_spatial_adata(adata_list, ncols=2, fixed_size=(2000, 2000)):
    """\
    Concatenate multiple anndata for visualization of spatial transcriptomics.

    Each sample's image is scaled to fit a tile of `fixed_size` pixels, keeping its
    aspect ratio, and the tiles are laid out in `ncols` columns. When every sample
    has `scalefactors["microns_per_pixel"]`, the samples share one scale, so their
    tissues keep their relative sizes; otherwise each fills its tile.

    The result uses the first sample's library id and `use_quality`, with the tiled
    image and `obsm["spatial"]` in full-resolution units of that image's scale
    factor, so `obsm["spatial"] * tissue_{use_quality}_scalef` lands on it.
    `spot_diameter_fullres` is the first sample's.

    Parameters
    ----------
    adata_list
        A list of anndata objects.
    ncols
        Number of columns
    fixed_size
        Size of each tile, in pixels: (height, width).
    Returns
    -------
    Returns adata.
    """

    use_adata_list = [adata.copy() for adata in adata_list]

    # check valid
    n_adata = len(use_adata_list)
    nrows = math.ceil(n_adata / ncols)
    if ncols > n_adata:
        raise ValueError("Number of column is out of bound")
    height, width = fixed_size

    samples = [_sample_image(adata) for adata in use_adata_list]
    # Microns per image pixel, where every sample has a micron scale.
    microns = [
        spatial["scalefactors"].get("microns_per_pixel") for _, spatial, _, _ in samples
    ]
    if all(m is not None for m in microns):
        microns_per_image_px = [
            m / scalef for m, (_, _, _, scalef) in zip(microns, samples, strict=True)
        ]
        # Microns per tiled-image pixel, so the largest tissue fills its tile.
        tiled_microns = max(
            max(image.shape[0] / height, image.shape[1] / width) * m
            for m, (_, _, image, _) in zip(microns_per_image_px, samples, strict=True)
        )
        resize_factors = [m / tiled_microns for m in microns_per_image_px]
    else:
        tiled_microns = None
        resize_factors = [
            min(height / image.shape[0], width / image.shape[1])
            for _, _, image, _ in samples
        ]

    # Each sample's image, resized and padded with white to its tile.
    tiles = []
    for (_, _, image, _), factor in zip(samples, resize_factors, strict=True):
        image = _as_float_rgb(image)
        shape = (
            min(height, max(1, round(image.shape[0] * factor))),
            min(width, max(1, round(image.shape[1] * factor))),
        )
        tile = np.ones((height, width, 3), dtype=np.float32)
        tile[: shape[0], : shape[1]] = resize(
            image, shape, order=1, anti_aliasing=True, preserve_range=True
        )
        tiles.append(tile)
    tiles += [np.ones((height, width, 3), dtype=np.float32)] * (nrows * ncols - n_adata)
    imgs_comb = np.vstack(
        [np.hstack(tiles[row * ncols : (row + 1) * ncols]) for row in range(nrows)]
    )

    # Coordinates: image pixels, resized and moved to the tile, then full-resolution
    # units of the first sample's scale factor.
    _, _, _, first_scalef = samples[0]
    for n, (adata, (_, _, _, scalef), factor) in enumerate(
        zip(use_adata_list, samples, resize_factors, strict=True)
    ):
        row, col = divmod(n, ncols)
        image_xy = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2] * scalef
        tiled_xy = image_xy * factor + np.array([width * col, height * row])
        adata.obs["imagecol"] = tiled_xy[:, 0]
        adata.obs["imagerow"] = tiled_xy[:, 1]
        adata.obsm["spatial"] = tiled_xy / first_scalef

    adata_concat = anndata.concat(use_adata_list, label="batch", index_unique="-")

    library_id, spatial, _, _ = samples[0]
    quality = spatial["use_quality"]
    scalefactors = {
        key: value
        for key, value in spatial["scalefactors"].items()
        if not (key.startswith("tissue_") and key.endswith("_scalef"))
    }
    if quality != "fulres":
        scalefactors[f"tissue_{quality}_scalef"] = first_scalef
    if "spot_diameter_fullres" in scalefactors:
        scalefactors["spot_diameter_fullres"] *= resize_factors[0]
    if tiled_microns is not None:
        scalefactors["microns_per_pixel"] = tiled_microns * first_scalef
    elif scalefactors.pop("microns_per_pixel", None) is not None:
        warnings.warn(
            "Not every sample has microns_per_pixel, so the samples are scaled to "
            "their tiles and the result has none.",
            stacklevel=2,
        )
    adata_concat.uns["spatial"] = {
        library_id: {
            **spatial,
            "images": {quality: as_uint8_image(imgs_comb)},
            "scalefactors": scalefactors,
        }
    }

    return adata_concat
