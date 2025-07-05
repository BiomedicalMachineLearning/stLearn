import zipfile as zf

import scanpy as sc
from anndata import AnnData
from scanpy.datasets._datasets import VisiumSampleID

from .._settings import settings

def visium_sge(
    sample_id: VisiumSampleID = "V1_Breast_Cancer_Block_A_Section_1",
    *,
    include_hires_tiff: bool = False,
) -> AnnData:
    """Processed Visium Spatial Gene Expression data from 10x Genomics’ database.

    The database_ can be browsed online to find the ``sample_id`` you want.

    .. _database: https://support.10xgenomics.com/spatial-gene-expression/datasets

    Parameters
    ----------
    sample_id
        The ID of the data sample in 10x’s spatial database.
    include_hires_tiff
        Download and include the high-resolution tissue image (tiff) in
        `adata.uns["spatial"][sample_id]["metadata"]["source_image_path"]`.

    Returns
    -------
    Annotated data matrix.
    """
    return sc.datasets.visium_sge(sample_id, include_hires_tiff=include_hires_tiff)

def xenium_sge(
    base_url="https://cf.10xgenomics.com/samples/xenium/1.0.1",
    image_filename="he_image.ome.tif",
    alignment_filename="he_imagealignment.csv",
    zip_filename="outs.zip",
    library_id="Xenium_FFPE_Human_Breast_Cancer_Rep1",
    include_hires_tiff: bool = False,
):
    """
    Download and extract Xenium SGE data files. Unlike scanpy this current does not
    load the data. Data is located in `settings.datasetdir` / `library_id`.

    Args:
        base_url: Base URL for downloads
        image_filename: Name of the image file to download
        alignment_filename: Name of the affine transformation file to download
        zip_filename: Name of the zip file to download
        library_id: Identifier for the library
        include_hires_tiff: Whether to download the high-res TIFF image
    """
    library_dir = settings.datasetdir / library_id
    library_dir.mkdir(parents=True, exist_ok=True)

    files_to_extract = ["cell_feature_matrix.h5", "cells.csv.gz", "experiment.xenium"]
    all_sge_files_exist = all(
        (library_dir / sge_file).exists() for sge_file in files_to_extract
    )

    download_filenames = []
    if not all_sge_files_exist:
        download_filenames.append(zip_filename)
    if include_hires_tiff and (
        not (library_dir / alignment_filename).exists()
        or not (library_dir / image_filename).exists()
    ):
        download_filenames += [alignment_filename, image_filename]

    for file_name in download_filenames:
        file_path = library_dir / file_name
        url = f"{base_url}/{library_id}/{library_id}_{file_name}"
        if not file_path.is_file():
            sc.readwrite._download(url=url, path=file_path)

    if not all_sge_files_exist:
        try:
            zip_file_path = library_dir / zip_filename
            with zf.ZipFile(zip_file_path, "r") as zip_ref:
                for zip_filename in files_to_extract:
                    with open(library_dir / zip_filename, "wb") as file_name:
                        file_name.write(zip_ref.read(f"outs/{zip_filename}"))
        except zf.BadZipFile:
            raise ValueError(f"Invalid zip file: {library_dir / zip_filename}")
