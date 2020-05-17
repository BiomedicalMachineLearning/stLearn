"""Reading and Writing
"""
from pathlib import Path, PurePath
from typing import Union

from anndata import AnnData
import numpy as np
from PIL import Image
import pandas as pd
import stlearn

def Read10X(
    count_matrix_file: Union[str, Path] = None,
    spatial_folder: Union[str, Path] = None,
    quality: str = "high",
    data_type: str = "h5"
    ) -> AnnData:
    
    if data_type == "h5":
        adata = stlearn.read.file_10x_h5(count_matrix_file)
        adata.var_names_make_unique()
    elif data_type == "mtx":
        adata = stlearn.read.file_10x_mtx(count_matrix_file)
    elif data_type == "table":
        adata = stlearn.read.file_table(count_matrix_file)
    else:
        raise ValueError("Wrong data type!")

    if quality == "low":
        stlearn.add.image(adata=adata, 
            imgpath=spatial_folder+"/tissue_lowres_image.png")
    elif quality == "high":
        stlearn.add.image(adata=adata, 
            imgpath=spatial_folder+"/tissue_hires_image.png")

    stlearn.add.positions(adata,
        position_filepath = spatial_folder+"/tissue_positions_list.csv",
        scale_filepath = spatial_folder+"/scalefactors_json.json",
            quality=quality)

    return adata


def ReadOldST(
    count_matrix_file: Union[str, Path] = None,
    spatial_file: Union[str, Path] = None,
    image_file: Union[str, Path] = None,
    ) -> AnnData:
    

    adata = stlearn.read.file_table(count_matrix_file)
    adata=stlearn.add.parsing(adata,
        coordinates_file = spatial_file)
    stlearn.add.image(adata, imgpath=image_file)

    return adata


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






