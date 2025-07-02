from pathlib import Path

import numpy as np
from anndata import AnnData
from PIL import Image
from tqdm import tqdm


def tiling(
    adata: AnnData,
    out_path: Path | str = "./tiling",
    library_id: str | None = None,
    crop_size: int = 40,
    target_size: int = 299,
    img_fmt: str = "JPEG",
    quality: int = 75,
    verbose: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """\
    Tiling H&E images to small tiles based on spot spatial location.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix containing spatial information.
    out_path: Path or str, default "./tiling"
        Path to save spot image tiles.
    library_id: str, optional
        Library id stored in AnnData. If None, uses first available library.
    crop_size: int, default 40
        Size of tiles to crop from original image.
    target_size: int, default 299
        Target size for resized tiles (input size for CNN).
    img_fmt: str, default "JPEG"
        Image format ('JPEG' or 'PNG').
    quality: int, default 75
        JPEG quality (1-100). Only used for JPEG format.
    verbose: bool, default False
        Enable verbose output.
    copy: bool, default False
        Return a copy instead of modifying adata in-place.

    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **tile_path** : `adata.obs` field
        Saved path for each spot image tiles
    """

    _validate_inputs(crop_size, target_size, img_fmt, quality)

    adata = adata.copy() if copy else adata

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    library_id = _get_library_id(adata, library_id)
    img_pillow = _load_and_prepare_image(adata, library_id)

    coordinates = list(zip(adata.obs["imagerow"], adata.obs["imagecol"]))

    tile_names = []

    with tqdm(
        total=len(adata),
        desc="Tiling image",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for image_row, image_col in coordinates:
            half_crop = crop_size // 2
            image_row_down = max(0, image_row - half_crop)
            image_row_up = image_row + half_crop
            image_col_left = max(0, image_col - half_crop)
            image_col_right = image_col + half_crop

            tile = img_pillow.crop(
                (image_col_left, image_row_down, image_col_right, image_row_up)
            )

            tile.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            tile = tile.resize((target_size, target_size))
            tile_name = str(image_col) + "-" + str(image_row) + "-" + str(crop_size)

            if img_fmt == "JPEG":
                out_tile = Path(out_path) / (tile_name + ".jpeg")
                tile_names.append(str(out_tile))
                tile.save(out_tile, "JPEG", quality=quality)
            else:
                out_tile = Path(out_path) / (tile_name + ".png")
                tile_names.append(str(out_tile))
                tile.save(out_tile, "PNG")

            if verbose:
                print(f"generate tile at location ({str(image_col)}, {str(image_row)})")

            pbar.update(1)

    adata.obs["tile_path"] = tile_names
    return adata if copy else None


def _validate_inputs(
    crop_size: int, target_size: int, img_fmt: str, quality: int
) -> None:

    if not isinstance(crop_size, int) or crop_size <= 0:
        raise ValueError("crop_size must be a positive integer")

    if not isinstance(target_size, int) or target_size <= 0:
        raise ValueError("target_size must be a positive integer")

    if img_fmt.upper() not in ["JPEG", "PNG"]:
        raise ValueError("img_fmt must be 'JPEG' or 'PNG'")

    if img_fmt.upper() == "JPEG" and (
        not isinstance(quality, int) or not 1 <= quality <= 100
    ):
        raise ValueError("quality must be an integer between 1 and 100 for JPEG format")


def _get_library_id(adata: AnnData, library_id: str | None) -> str:
    if library_id is None:
        try:
            library_id = list(adata.uns["spatial"].keys())[0]
        except (KeyError, IndexError):
            raise ValueError("No spatial data found in adata.uns['spatial']")

    if library_id not in adata.uns["spatial"]:
        raise ValueError(f"Library '{library_id}' not found in spatial data")

    return library_id


def _load_and_prepare_image(adata: AnnData, library_id: str) -> Image.Image:
    try:
        spatial_data = adata.uns["spatial"][library_id]
        use_quality = spatial_data["use_quality"]
        image = spatial_data["images"][use_quality]
    except KeyError as e:
        raise ValueError(
            f"Could not find image data in adata.uns['spatial']['{library_id}']: {e}"
        )

    if image.dtype in (np.float32, np.float64):
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)

    img_pillow = Image.fromarray(image)

    if img_pillow.mode == "RGBA":
        img_pillow = img_pillow.convert("RGB")

    return img_pillow
