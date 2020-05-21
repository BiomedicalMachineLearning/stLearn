"""Reading and Writing
"""
from pathlib import Path, PurePath
from typing import Optional, Union
from anndata import AnnData
import numpy as np
from PIL import Image
import pandas as pd
import stlearn
import scanpy

def Read10X(
    path: Union[str, Path],
    genome: Optional[str] = None,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str = None,
    load_images: Optional[bool] = True,
    use_quality: str = "hires"
    ) -> AnnData:
    
    from scanpy import read_visium
    adata = read_visium(path, genome=None,
     count_file='filtered_feature_bc_matrix.h5',
      library_id=None,
       load_images=True)
    adata.var_names_make_unique()

    
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
        
    scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_"+use_quality+"_scalef"]
    image_coor = adata.obsm["spatial"]*scale

    adata.obs["imagecol"] = image_coor[:,0]
    adata.obs["imagerow"] = image_coor[:,1]
    adata.uns["spatial"]["use_quality"] = use_quality

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
    library_id: str = None,
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

    adata.uns["spatial"] = {}
    adata.uns["spatial"]["Slide-seq"] = {}
    adata.uns["spatial"]["Slide-seq"]["images"] = {}
    adata.uns["spatial"]["Slide-seq"]["images"]["hires"] = imgarr
    adata.uns["spatial"]["use_quality"] = "hires"
    adata.uns["spatial"]["Slide-seq"]["scalefactors"] = {}
    adata.uns["spatial"]["Slide-seq"]["scalefactors"]["tissue_hires_scalef"] = 1

    adata.obsm["spatial"] = meta[["x","y"]].values

    return adata






