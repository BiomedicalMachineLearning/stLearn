"""Reading and Writing
"""
from pathlib import Path, PurePath
from typing import Union

from anndata import AnnData
import numpy as np
from PIL import Image
import pandas as pd

def ReadSlideSeq(
    count_matrix_file: Union[str, Path],
    spatial_file: Union[str, Path],
    scale_factor: float = 10,
) -> AnnData:

    """\
    Read Slide-seq data

    Parameters
    ----------
    count_matrix_file
        Path to count matrix file.
    spatial_file
        Path to spatial location file.
    Returns
    -------
    AnnData
    """

    count = pd.read_csv(count_matrix_file)
    meta = pd.read_csv(spatial_file)

    adata = AnnData(count.iloc[:,1:].set_index("gene").T)

    adata.var["ENSEMBL"] = count["ENSEMBL"].values

    adata.obs["index"] = meta["index"].values
    adata.obs["imagecol"] = meta["x"].values/scale_factor
    adata.obs["imagerow"] = meta["y"].values/scale_factor

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(),adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1*max_size)

    image = Image.new('RGB', (max_size, max_size), (255, 255, 255))
    imgarr = np.array(image) 
    adata.uns["tissue_img"] = imgarr

    return adata






