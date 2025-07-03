import zipfile as zf

import scanpy as sc
from anndata import AnnData

from .._settings import settings


def example_bcba() -> AnnData:
    """\
    Download processed BCBA data (10X genomics published data).
    Reference:
    https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1
    """
    settings.datasetdir.mkdir(parents=True, exist_ok=True)
    filename = settings.datasetdir / "example_bcba.h5"
    url = "https://www.dropbox.com/s/u3m2f16mvdom1am/example_bcba.h5ad?dl=1"
    if not filename.is_file():
        sc.readwrite._download(url=url, path=filename)
    adata = sc.read_h5ad(filename)
    return adata


def xenium_sge(
        base_url="https://cf.10xgenomics.com/samples/xenium/1.0.1",
        image_filename="he_image.ome.tif",
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
    if include_hires_tiff and not (library_dir / image_filename).exists():
        download_filenames += ["imagealignment.csv", image_filename]

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
