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
            adata.uns["spatial"] = {}
            adata.uns["spatial"]["Slide-seq"] = {}
            adata.uns["spatial"]["Slide-seq"]["images"] = {}
            adata.uns["spatial"]["Slide-seq"]["images"]["hires"] = img
            adata.uns["spatial"]["use_quality"] = "hires"
            adata.uns["spatial"]["Slide-seq"]["scalefactors"] = {}
            adata.uns["spatial"]["Slide-seq"]["scalefactors"]["tissue_hires_scalef"] = 1

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
