import os
from pathlib import Path

import numpy as np
from anndata import AnnData
from PIL import Image

# Test progress bar
from tqdm import tqdm


def tiling(
    adata: AnnData,
    out_path: Path | str = "./tiling",
    library_id: str | None = None,
    crop_size: int = 40,
    target_size: int = 299,
    img_fmt: str = "JPEG",
    quality: int = 95,
    verbose: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """\
    Tiling H&E images to small tiles based on spot spatial location

    Parameters
    ----------
    adata:
        Annotated data matrix.
    out_path:
        Path to save spot image tiles
    library_id:
        Library id stored in AnnData.
    crop_size:
        Size of tiles
    target_size:
        Input size for convolutional neuron network
    img_fmt:
        Image format ('JPEG' or 'PNG')
    quality:
        JPEG quality 1-100.
    verbose:
        Verbose output
    copy:
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **tile_path** : `adata.obs` field
        Saved path for each spot image tiles
    """

    adata = adata.copy() if copy else adata

    if not isinstance(crop_size, int) or crop_size <= 0:
        raise ValueError("crop_size must be a positive integer")
    if not isinstance(target_size, int) or target_size <= 0:
        raise ValueError("target_size must be a positive integer")
    if img_fmt.upper() not in ["JPEG", "PNG"]:
        raise ValueError("img_fmt must be 'JPEG' or 'PNG'")

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load and prepare image
    try:
        image = adata.uns["spatial"][library_id]["images"][
            adata.uns["spatial"][library_id]["use_quality"]
        ]
    except KeyError as e:
        raise ValueError(f"Could not find image data in adata.uns['spatial']: {e}")

    if image.dtype in (np.float32, np.float64):
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)

    img_pillow = Image.fromarray(image)

    if img_pillow.mode == "RGBA":
        img_pillow = img_pillow.convert("RGB")

    coordinates = list(zip(adata.obs["imagerow"], adata.obs["imagecol"]))

    tile_names = []

    with tqdm(
        total=len(adata),
        desc="Tiling image",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for imagerow, imagecol in coordinates:
            half_crop = crop_size // 2
            imagerow_down = max(0, imagerow - half_crop)
            imagerow_up = imagerow + half_crop
            imagecol_left = max(0, imagecol - half_crop)
            imagecol_right = imagecol + half_crop

            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up)
            )

            tile.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            tile = tile.resize((target_size, target_size))
            tile_name = str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)

            if img_fmt == "JPEG":
                out_tile = Path(out_path) / (tile_name + ".jpeg")
                tile_names.append(str(out_tile))
                tile.save(out_tile, "JPEG", quality=quality)
            else:
                out_tile = Path(out_path) / (tile_name + ".png")
                tile_names.append(str(out_tile))
                tile.save(out_tile, "PNG")

            if verbose:
                print(f"generate tile at location ({str(imagecol)}, {str(imagerow)})")

            pbar.update(1)

    adata.obs["tile_path"] = tile_names
    return adata if copy else None
