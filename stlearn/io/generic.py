"""Reading and Writing"""

import pathlib

import anndata
import numpy
import pandas
import PIL.Image as PIL_Image

import stlearn
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
        Pandas Dataframe of spatial location of cells/spots.
    library_id
        Identifier for the library. Can be modified when concatenating multiple
        adata objects.
    image_path:
        Path to image. Only need when loading full resolution image.
    scale
        Set scale factor.
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

    adata.obsm["spatial"] = spatial.values

    if scale is None:
        max_coor = numpy.max(adata.obsm["spatial"])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = spatial["imagecol"].values * scale
    adata.obs["imagerow"] = spatial["imagerow"].values * scale

    if image_path is not None:
        stlearn.add.image(
            adata,
            library_id=library_id,
            quality=quality,
            imgpath=image_path,
            scale=scale,
            spot_diameter_fullres=spot_diameter_fullres,
        )
    else:
        # Create image
        max_size = numpy.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
        max_size = int(max_size + 0.1 * max_size)

        if background_color == "black":
            image = PIL_Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
        else:
            image = PIL_Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
        imgarr = numpy.array(image)

        # Create spatial dictionary
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
