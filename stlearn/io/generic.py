"""Reading and Writing"""

import pathlib

import anndata
import numpy
import pandas
import PIL.Image as PIL_Image

from stlearn.adds._barcodes import by_barcode
from stlearn.adds.image import _read_image_file
from stlearn.types import _BACKGROUND


def create_stlearn(
    count: pandas.DataFrame,
    spatial: pandas.DataFrame,
    library_id: str,
    image_path: pathlib.Path | None = None,
    scale: float | None = None,
    quality: str = "hires",
    spot_diameter_fullres: float = 50,
    background_color: _BACKGROUND = "white",
) -> anndata.AnnData:
    """\
    Create AnnData object for stLearn

    Parameters
    ----------
    count
        Pandas Dataframe of count matrix with rows as barcodes and columns as gene names
    spatial
        Pandas Dataframe of spatial location of cells/spots, with `imagecol` and
        `imagerow` columns, indexed by barcode. Its rows are matched to `count`'s
        by barcode; spots without a row get NaN, with a warning.
    library_id
        Identifier for the library. Can be modified when concatenating multiple
        adata objects.
    image_path:
        Path to a tissue image to show behind the spots. Without one, a blank
        image is made.
    scale
        Image pixels per unit of `spatial`'s coordinates: `obs['imagecol']` and
        `obs['imagerow']` are the coordinates times `scale`, while
        `obsm['spatial']` keeps them as given. If None, 1 with an image (the
        coordinates are its pixels), otherwise 2000 / the largest coordinate.
    quality
        Set quality that convert to stlearn to use. Store in
        anndata.obs['imagecol' & 'imagerow']
    spot_diameter_fullres
        Diameter of spot in full resolution
    background_color
        Color of the background. Only `black` or `white` is allowed.
    Returns
    -------
    AnnData
    """
    adata = anndata.AnnData(X=count)
    adata.X = adata.X.copy()
    spatial = by_barcode(spatial, adata, "spatial")
    adata.obsm["spatial"] = spatial.values

    if scale is None and image_path is not None:
        # Pixels equal coordinates of the image.
        scale = 1.0
    elif scale is None:
        # Fit the spots to a blank image about 2000 pixels across.
        max_coor = numpy.nanmax(adata.obsm["spatial"])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = spatial["imagecol"].values * scale
    adata.obs["imagerow"] = spatial["imagerow"].values * scale

    if image_path is not None:
        imgarr = _read_image_file(image_path)
    else:
        # Create a plain black / white image.
        max_size = numpy.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
        max_size = int(max_size + 0.1 * max_size)
        colour = (0, 0, 0, 0) if background_color == "black" else (255, 255, 255, 255)
        image = PIL_Image.new("RGBA", (max_size, max_size), colour)
        imgarr = numpy.array(image)

    # Create a spatial dictionary
    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"][
        "tissue_" + quality + "_scalef"
    ] = scale
    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"
    ] = spot_diameter_fullres

    return adata
