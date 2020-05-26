from typing import Optional, Union
from anndata import AnnData
from matplotlib import pyplot as plt
from pathlib import Path
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def image(
    adata: AnnData,
    imgpath: Union[Path, str],
    library_id: str,
    quality: str = "hires",
    scale: float = 1.0,
    visium: bool = False,
    copy: bool = False,
) -> Optional[AnnData]:

    """\
    Adding image data to the Anndata object

    Parameters
    ----------
    adata
        Annotated data matrix.
    imgpath
        Image path.
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    scale
        Set scale factor.
    quality
        Set quality that convert to stlearn to use. Store in anndata.obs['imagecol' & 'imagerow'].
    visium
        Is this anndata read from Visium platform or not.
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **tissue_img** : `adata.uns` field
        Array format of image, saving by Pillow package.
    """
    

    if imgpath is not None and os.path.isfile(imgpath):
        try:
            img = plt.imread(imgpath, 0)

            if visium:
                adata.uns["spatial"][library_id]["images"][quality] = img
            else:
                adata.uns["spatial"] = {}
                adata.uns["spatial"][library_id] = {}
                adata.uns["spatial"][library_id]["images"] = {}
                adata.uns["spatial"][library_id]["images"][quality] = img
                adata.uns["spatial"]["use_quality"] = quality
                adata.uns["spatial"][library_id]["scalefactors"] = {}
                adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale

            print("Added tissue image to the object!")

            return adata if copy else None
        except:
            raise ValueError(f'''\
            {imgpath!r} does not end on a valid extension.
            ''')
    else:
        raise ValueError(f'''\
        {imgpath!r} does not end on a valid extension.
        ''')
    return adata if copy else None
