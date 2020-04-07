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
    if imgpath is not None and os.path.isfile(imgpath):
        try:
            img = plt.imread(imgpath, 0)
            adata.uns["tissue_img"] = img
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
