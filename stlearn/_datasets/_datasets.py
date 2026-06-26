import zipfile as zf

import scanpy as sc
from anndata import AnnData

from .._settings import settings


def visium_sge(
    sample_id="V1_Breast_Cancer_Block_A_Section_1",
    *,
    include_hires_tiff: bool = False,
) -> AnnData:
    """Processed Visium Spatial Gene Expression data from 10x Genomics' database.

    The database_ can be browsed online to find the ``sample_id`` you want.

    .. _database: https://support.10xgenomics.com/spatial-gene-expression/datasets

    Parameters
    ----------
    sample_id
        The ID of the data sample in 10x's spatial database.
    include_hires_tiff
        Download and include the high-resolution tissue image (tiff) in
        `adata.uns["spatial"][sample_id]["metadata"]["source_image_path"]`.

    Returns
    -------
    Annotated data matrix.
    """
    sc.settings.datasetdir = settings.datasetdir
    return sc.datasets.visium_sge(sample_id, include_hires_tiff=include_hires_tiff)


def xenium_sge(
    base_url: str="https://cf.10xgenomics.com/samples/xenium/1.0.1",
    library_id: str="Xenium_FFPE_Human_Breast_Cancer_Rep1",
    zip_filename: str="outs.zip",
    image_filename: str="he_image.ome.tif",
    alignment_filename: str="he_imagealignment.csv",
    include_hires_tiff: bool = False,
):
    """
    Download and extract Xenium SGE data files. Unlike scanpy this current does not
    load the data. Data is located in `settings.datasetdir` / `library_id`.

    Args:
        base_url: Base URL for downloads
        library_id: Identifier for the library
        zip_filename: Name of the zip file to download
        image_filename: Name of the image file to download
        alignment_filename: Name of the affine transformation file to download
        include_hires_tiff: Whether to download the high-res TIFF image
    """
    sc.settings.datasetdir = settings.datasetdir
    library_dir = settings.datasetdir / library_id
    library_dir.mkdir(parents=True, exist_ok=True)

    if "xe_outs.zip" in zip_filename:
        files_to_extract = [
            "cell_feature_matrix.zarr.zip", "cells.zarr.zip", "experiment.xenium"
        ]
    else:
        files_to_extract = [
            "cell_feature_matrix.h5", "cells.csv.gz", "experiment.xenium"
        ]

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
        zip_file_path = library_dir / zip_filename
        try:
            with zf.ZipFile(zip_file_path, "r") as zip_ref:
                members = {m.rsplit("/", 1)[-1]: m for m in zip_ref.namelist()}
                for name in files_to_extract:
                    (library_dir / name).write_bytes(zip_ref.read(members[name]))
        except zf.BadZipFile as b:
            raise ValueError(f"Invalid zip file: {zip_file_path}") from b
