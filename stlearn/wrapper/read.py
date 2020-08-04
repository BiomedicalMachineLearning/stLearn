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
import scipy

def Read10X(
    path: Union[str, Path],
    genome: Optional[str] = None,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str = None,
    load_images: Optional[bool] = True,
    quality: str = "hires"
    ) -> AnnData:
    
    """\
    Read Visium data from 10X (wrap read_visium from scanpy)

    In addition to reading regular 10x output,
    this looks for the `spatial` folder and loads images,
    coordinates and scale factors.
    Based on the `Space Ranger output docs`_.

    .. _Space Ranger output docs: https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview

    Parameters
    ----------
    path
        Path to directory for visium datafiles.
    genome
        Filter expression to genes within this genome.
    count_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        'filtered_feature_bc_matrix.h5' or 'raw_feature_bc_matrix.h5'.
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    load_images
        Load image or not.
    quality
        Set quality that convert to stlearn to use. Store in anndata.obs['imagecol' & 'imagerow']


    Returns
    -------
    Annotated data matrix, where observations/cells are named by their
    barcode and variables/genes by gene name. Stores the following information:
    :attr:`~anndata.AnnData.X`
        The data matrix is stored
    :attr:`~anndata.AnnData.obs_names`
        Cell names
    :attr:`~anndata.AnnData.var_names`
        Gene names
    :attr:`~anndata.AnnData.var`\\ `['gene_ids']`
        Gene IDs
    :attr:`~anndata.AnnData.var`\\ `['feature_types']`
        Feature types
    :attr:`~anndata.AnnData.uns`\\ `['spatial']`
        Dict of spaceranger output files with 'library_id' as key
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['images']`
        Dict of images (`'hires'` and `'lowres'`)
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['scalefactors']`
        Scale factors for the spots
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['metadata']`
        Files metadata: 'chemistry_description', 'software_version'
    :attr:`~anndata.AnnData.obsm`\\ `['spatial']`
        Spatial spot coordinates, usable as `basis` by :func:`~scanpy.pl.embedding`.
    """
    
    from scanpy import read_visium
    adata = read_visium(path, genome=None,
     count_file=count_file,
      library_id=None,
       load_images=True)
    adata.var_names_make_unique()

    adata.obs['sum_counts'] = np.array(adata.X.sum(axis=1))

    
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
        
    scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_"+quality+"_scalef"]
    image_coor = adata.obsm["spatial"]*scale

    adata.obs["imagecol"] = image_coor[:,0]
    adata.obs["imagerow"] = image_coor[:,1]
    adata.uns["spatial"]["use_quality"] = quality

    return adata


def ReadOldST(
    count_matrix_file: Union[str, Path] = None,
    spatial_file: Union[str, Path] = None,
    image_file: Union[str, Path] = None,
    library_id: str = "OldST",
    scale: float = 1.0,
    quality: str = "hires"
    ) -> AnnData:

    """\
    Read Old Spatial Transcriptomics data

    Parameters
    ----------
    count_matrix_file
        Path to count matrix file.
    spatial_file
        Path to spatial location file.
    image_file
        Path to the tissue image file
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    scale
        Set scale factor.
    quality
        Set quality that convert to stlearn to use. Store in anndata.obs['imagecol' & 'imagerow']
    Returns
    -------
    AnnData
    """

    adata = stlearn.read.file_table(count_matrix_file)
    adata=stlearn.add.parsing(adata,
        coordinates_file = spatial_file)
    stlearn.add.image(adata, library_id=library_id, quality=quality,imgpath=image_file, scale=scale)

    adata.obs['sum_counts'] = np.array(adata.X.sum(axis=1))

    return adata


def ReadSlideSeq(
    count_matrix_file: Union[str, Path],
    spatial_file: Union[str, Path],
    library_id: str = None,
    scale: float = 1.0,
    quality: str = "hires"
    ) -> AnnData:

    """\
    Read Slide-seq data

    Parameters
    ----------
    count_matrix_file
        Path to count matrix file.
    spatial_file
        Path to spatial location file.
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    scale
        Set scale factor.
    quality
        Set quality that convert to stlearn to use. Store in anndata.obs['imagecol' & 'imagerow']
    Returns
    -------
    AnnData
    """

    count = pd.read_csv(count_matrix_file)
    meta = pd.read_csv(spatial_file)

    adata = AnnData(count.iloc[:,1:].set_index("gene").T)

    adata.var["ENSEMBL"] = count["ENSEMBL"].values

    adata.obs["index"] = meta["index"].values
    
    adata.obs["imagecol"] = meta["x"].values*scale
    adata.obs["imagerow"] = meta["y"].values*scale

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(),adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1*max_size)

    image = Image.new('RGB', (max_size, max_size), (0, 0, 0))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "Slide-seq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.obsm["spatial"] = meta[["x","y"]].values

    adata.obs['sum_counts'] = np.array(adata.X.sum(axis=1))
    

    return adata

def ReadMERFISH(
    count_matrix_file: Union[str, Path],
    spatial_file: Union[str, Path],
    library_id: str = None,
    scale: float = 1,
    quality: str = "hires"
    ) -> AnnData:

    """\
    Read MERFISH data

    Parameters
    ----------
    count_matrix_file
        Path to count matrix file.
    spatial_file
        Path to spatial location file.
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    scale
        Set scale factor.
    quality
        Set quality that convert to stlearn to use. Store in anndata.obs['imagecol' & 'imagerow']
    Returns
    -------
    AnnData
    """

    coordinates = pd.read_excel(spatial_file, index_col=0)
    if coordinates.min().min() < 0:
        coordinates = coordinates + np.abs(coordinates.min().min())+100
    from scanpy import read_csv
    counts = read_csv(count_matrix_file).transpose()

    adata_merfish = counts[coordinates.index, :]
    adata_merfish.obsm["spatial"] = coordinates.to_numpy()
    adata_merfish.obs["imagecol"] = adata_merfish.obsm["spatial"][:,0]
    adata_merfish.obs["imagerow"] = adata_merfish.obsm["spatial"][:,1]

    # Create image
    max_size = np.max([adata_merfish.obs["imagecol"].max(),adata_merfish.obs["imagerow"].max()])
    max_size = int(max_size + 0.1*max_size)
    image = Image.new('RGB', (max_size, max_size), (255, 255, 255))
    imgarr = np.array(image)


    if library_id is None:
        library_id = "MERSEQ"

    adata_merfish.uns["spatial"] = {}
    adata_merfish.uns["spatial"][library_id] = {}
    adata_merfish.uns["spatial"][library_id]["images"] = {}
    adata_merfish.uns["spatial"][library_id]["images"][quality] = imgarr
    adata_merfish.uns["spatial"]["use_quality"] = quality
    adata_merfish.uns["spatial"][library_id]["scalefactors"] = {}
    adata_merfish.uns["spatial"][library_id]["scalefactors"]["tissue_"+quality+"_scalef"] = scale

    adata_merfish.obs["imagecol"] = adata_merfish.obsm["spatial"][:,0]*scale
    adata_merfish.obs["imagerow"] = adata_merfish.obsm["spatial"][:,1]*scale

    adata_merfish.obs['sum_counts'] = np.array(adata_merfish.X.sum(axis=1))

    return adata_merfish

def ReadSeqFish(
    count_matrix_file: Union[str, Path],
    spatial_file: Union[str, Path],
    library_id: str = None,
    scale: float = 1.0,
    quality: str = "hires",
    field: int = 0
    ) -> AnnData:

    """\
    Read SeqFish data

    Parameters
    ----------
    count_matrix_file
        Path to count matrix file.
    spatial_file
        Path to spatial location file.
    library_id
        Identifier for the visium library. Can be modified when concatenating multiple adata objects.
    scale
        Set scale factor.
    quality
        Set quality that convert to stlearn to use. Store in anndata.obs['imagecol' & 'imagerow']
    field
        Set field of view for SeqFish data
    Returns
    -------
    AnnData
    """

    count = pd.read_table(count_matrix_file,header=None)
    spatial = pd.read_table(spatial_file,index_col=False)
    
    count = count.T
    count.columns = count.iloc[0]
    count = count.drop(count.index[0]).reset_index(drop=True)
    count = count[count["Field_of_View"] == field].drop(count.columns[[0,1]],axis=1)
    
    spatial = spatial[spatial["Field_of_View"] == field]
    
    adata = AnnData(count)
    
    adata.obs["imagecol"] = spatial["X"].values*scale
    adata.obs["imagerow"] = spatial["Y"].values*scale

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(),adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1*max_size)

    image = Image.new('RGB', (max_size, max_size), (255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "Slide-seq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.obsm["spatial"] = spatial[["X","Y"]].values

    adata.obs['sum_counts'] = np.array(adata.X.sum(axis=1))
    

    return adata