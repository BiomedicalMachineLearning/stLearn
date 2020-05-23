from typing import Optional, Union
from anndata import AnnData
from .._compat import Literal
from PIL import Image
from pathlib import Path
# Test progress bar
from tqdm import tqdm
import numpy as np

def tiling(
        adata: AnnData,
        out_path: Union[Path, str] = "./",
        library_id: str = None,
        crop_size: int = 40,
        target_size: int = 299,
        verbose: bool = False,
        copy: bool = False
) -> Optional[AnnData]:
    """\
    Tiling H&E images to small tiles based on spot spatial location

    Parameters
    ----------
    adata
        Annotated data matrix.
    out_path
        Path to save spot image tiles
    library_id
        Library id stored in AnnData.
    crop_size
        Size of tiles
    verbose
        Verbose output
    copy
        Return a copy instead of writing to adata.
    target_size
        Input size for convolutional neuron network
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **tile_path** : `adata.obs` field
        Saved path for each spot image tiles
    """

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    image = (adata.uns["spatial"][library_id]["images"][adata.uns["spatial"]["use_quality"]]* 255).astype(np.uint8)

    img_pillow = Image.fromarray(image)
    tile_names = []

    with tqdm(total=len(adata), desc="Tiling image", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up))
            tile.thumbnail((target_size, target_size), Image.ANTIALIAS)
            tile.resize((target_size, target_size))
            tile_name = str(imagecol) + '-' + str(imagerow) + \
                '-' + str(crop_size)
            out_tile = Path(out_path) / (tile_name + '.jpeg')
            tile_names.append(str(out_tile))
            if verbose:
                print("generate tile at location ({}, {})".format(
                    str(imagecol), str(imagerow)))
            tile.save(out_tile, 'JPEG')

            pbar.update(1)

    adata.obs["tile_path"] = tile_names
    return adata if copy else None
