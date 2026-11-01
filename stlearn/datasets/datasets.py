import pathlib
import shutil

import pooch
import requests
import spatialdata
import spatialdata_io

from .._settings import settings
from ..io.spatialdata import visium_hd, xenium

VISIUM_BASE_URL = "https://cf.10xgenomics.com/samples/spatial-exp"
XENIUM_BASE_URL = "https://cf.10xgenomics.com/samples/xenium"
ATERA_BASE_URL = "https://cf.10xgenomics.com/samples/atera"

# CDN and S3 buckets
_CDN_URL = "https://cf.10xgenomics.com/"
_S3_URL = "https://s3-us-west-2.amazonaws.com/10x.files/"

XENIUM_SAMPLES = {
    "Xenium_FFPE_Human_Breast_Cancer_Rep1": "1.0.1",
    "Xenium_FFPE_Human_Breast_Cancer_Rep2": "1.0.1",
    "Xenium_V1_hKidney_nondiseased_section": "1.5.0",
    "Xenium_V1_hKidney_cancer_section": "1.5.0",
    "Xenium_V1_hPancreas_Cancer_Add_on_FFPE": "1.6.0",
    "Xenium_V1_hColon_Non_diseased_Base_FFPE": "1.6.0",
    "Xenium_V1_hColon_Cancer_Add_on_FFPE": "1.6.0",
    "Xenium_V1_hSkin_Melanoma_Base_FFPE": "1.6.0",
    "Xenium_V1_mouse_pup": "1.6.0",
    "Xeniumranger_V1_hSkin_Melanoma_Add_on_FFPE": "1.7.0",
    "Xenium_V1_hSkin_nondiseased_section_1_FFPE": "1.9.0",
    "Xenium_V1_hSkin_nondiseased_section_2_FFPE": "1.9.0",
    "Xenium_V1_hHeart_nondiseased_section_FFPE": "1.9.0",
    "Xenium_V1_hLiver_nondiseased_section_FFPE": "1.9.0",
    "Xenium_V1_hLiver_cancer_section_FFPE": "1.9.0",
    "Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE": "1.9.0",
    "Xenium_V1_hTonsil_follicular_lymphoid_hyperplasia_section_FFPE": "1.9.0",
    "Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section": "1.9.0",
    "Xenium_V1_hBoneMarrow_nondiseased_section": "1.9.0",
    "Xenium_V1_hBone_nondiseased_section": "1.9.0",
    "Xenium_V1_mFemur_formic_acid_24hrdecal_section": "1.9.0",
    "Xenium_V1_mFemur_EDTA_3daydecal_section": "1.9.0",
    "Xenium_V1_mFemur_EDTA_PFA_3daydecal_section": "1.9.0",
    "Xenium_V1_humanLung_Cancer_FFPE": "2.0.0",
    "Xenium_V1_human_Pancreas_FFPE": "2.0.0",
    "Xenium_V1_mouse_Colon_FF": "2.0.0",
    "Xenium_V1_Human_Colorectal_Cancer_Addon_FFPE": "2.0.0",
    "Xenium_V1_Human_Ovarian_Cancer_Addon_FFPE": "2.0.0",
    "Xenium_V1_Human_Lung_Cancer_Addon_FFPE": "2.0.0",
    "Xenium_V1_Human_Brain_GBM_FFPE": "2.0.0",
    "Xenium_V1_Human_Ductal_Adenocarcinoma_FFPE": "2.0.0",
    "Xenium_Prime_Breast_Cancer_FFPE": "3.0.0",
    "Xenium_Prime_Cervical_Cancer_FFPE": "3.0.0",
    "Xenium_Prime_Human_Lung_Cancer_FFPE": "3.0.0",
    "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE": "3.0.0",
    "Xenium_Prime_Human_Ovary_FF": "3.0.0",
    "Xenium_Prime_Human_Prostate_FFPE": "3.0.0",
    "Xenium_Prime_Human_Skin_FFPE": "3.0.0",
    "Xenium_Prime_Mouse_Brain_Coronal_FF": "3.0.0",
    "Xenium_Prime_Mouse_Pup_FFPE": "3.0.0",
    "Xenium_Prime_Ovarian_Cancer_FFPE": "3.0.0",
    "Xenium_Prime_Ovarian_Cancer_FFPE_XRrun": "3.0.0",
    "Xenium_V1_Human_Lung_Cancer_FFPE": "3.0.0",
    "Xenium_V1_Human_Kidney_FFPE_Protein": "4.0.0",
    "Xenium_V1_Human_Ovary_Cancer_FF": "4.0.0",
    "Xenium_Prime_Human_Ovary_Cancer_FF": "4.0.0",
}

ATERA_SAMPLES = {
    "WTA_Preview_FFPE_Breast_Cancer": "dev",
    "WTA_Preview_FFPE_Cervical_Cancer": "dev",
}

VISIUM_HD_SAMPLES = {
    "Visium_HD_Human_Colon_Cancer": "3.0.0",  # also 4.0.1
    "Visium_HD_Human_Colon_Cancer_P1": "3.0.0",
    "Visium_HD_Human_Colon_Cancer_P2": "3.0.0",
    "Visium_HD_Human_Colon_Cancer_P5": "3.0.0",
    "Visium_HD_Human_Colon_Normal_P3": "3.0.0",
    "Visium_HD_Human_Colon_Normal_P5": "3.0.0",
    "Visium_HD_Human_Lung_Cancer": "3.0.0",
    "Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1": "3.0.0",
    "Visium_HD_Human_Lung_Cancer_HD_Only_Experiment2": "3.0.0",
    "Visium_HD_Human_Lung_Cancer_post_Xenium_v1_Experiment1": "3.0.0",
    "Visium_HD_Human_Lung_Cancer_post_Xenium_Prime_5K_Experiment2": "3.0.0",
    "Visium_HD_Human_Pancreas": "3.0.0",  # also 4.0.1
    "Visium_HD_Mouse_Brain": "3.0.0",  # also 4.0.1
    "Visium_HD_Mouse_Embryo": "3.0.0",  # also 4.0.1
    "Visium_HD_Mouse_Kidney": "3.0.0",  # also 4.0.1
    "Visium_HD_Mouse_Small_Intestine": "3.0.0",  # also 4.0.1
    "Visium_HD_Human_Breast_Cancer_Fresh_Frozen": "3.1.1",
    "Visium_HD_Human_Breast_Cancer_Fixed_Frozen": "3.1.1",
    "Visium_HD_Human_Lung_Cancer_Fixed_Frozen": "3.1.1",
    "Visium_HD_Human_Tonsil_Fresh_Frozen": "3.1.1",
    "Visium_HD_Human_Tonsil_Fresh_Frozen_IF": "3.1.1",
    "Visium_HD_Mouse_Brain_Fresh_Frozen": "3.1.1",
    "Visium_HD_Mouse_Brain_Fixed_Frozen": "3.1.1",
    "Visium_HD_Mouse_Lung_Fresh_Frozen": "3.1.1",
    "Visium_HD_FF_Human_Breast_Cancer": "3.1.2",  # also 4.0.1
    "Visium_HD_FF_Human_Tonsil": "3.1.2",
    "Visium_HD_Human_Breast_Cancer_FFPE": "3.1.2",
    "Visium_HD_Human_Kidney_FFPE": "3.1.2",  # also 4.0.1
    "Visium_HD_Human_Lymph_Node_FFPE": "4.0.1",
    "Visium_HD_Human_Prostate_Cancer_FFPE": "4.0.1",
    "Visium_HD_Human_Ovarian_Cancer_FF": "4.0.1",
    "Visium_HD_Human_Ovarian_Cancer_FF_Min_Depth": "4.0.1",
    "Visium_HD_3prime_Arabidopsis": "4.0.1",
    "Visium_HD_3prime_Human_Mouse_Xenograft": "4.0.1",
    "Visium_HD_3prime_Human_Ovarian_Cancer": "4.0.1",
    "Visium_HD_3prime_Human_Ovarian_Cancer_FF": "4.0.1",
    "Visium_HD_3prime_Human_Ovarian_Cancer_FF_Min_Depth": "4.0.1",
    "Visium_HD_3prime_Human_Pancreatic_Cancer": "4.0.1",
    "Visium_HD_3prime_Mouse_Brain": "4.0.1",
    "Visium_HD_3prime_Mouse_Embryo": "4.0.1",
    "Visium_HD_3prime_Mouse_Kidney": "4.0.1",
    "Visium_HD_3prime_Rhesus_Kidney": "4.0.1",
    "Visium_HD_3prime_Soybean": "4.0.1",
    "Visium_HD_3prime_Zebrafish_Head": "4.0.1",
    "Visium_HD_3prime_Human_Lymph_Node": "4.1.0",
    "Visium_HD_3prime_Human_Tonsil": "4.1.0",
    "Visium_HD_3prime_Mouse_Ovary": "4.1.0",
    "Visium_HD_3prime_Rhesus_Liver": "4.1.0",
    "Visium_HD_6p5mm_Human_Brain_Cancer": "4.1.0",
    "Visium_HD_6p5mm_Human_Colon_Cancer": "4.1.0",
    "Visium_HD_6p5mm_Human_Heart": "4.1.0",
    "Visium_HD_6p5mm_Mouse_Brain": "4.1.0",
    "Visium_HD_6p5mm_Rat_Liver": "4.1.0",
    "Visium_HD_11mm_Human_Breast_Cancer": "4.1.0",
    "Visium_HD_11mm_Human_Breast_Cancer_TMA": "4.1.0",
    "Visium_HD_11mm_Human_Colon_Cancer_Fresh_Frozen": "4.1.0",
    "Visium_HD_11mm_Human_Pancreas": "4.1.0",
    "Visium_HD_11mm_Human_TA": "4.1.0",
    "Visium_HD_11mm_Mouse_Embryo": "4.1.0",
}

# Samples whose full-resolution image is tissue_image.tif; the rest use .btf.
VISIUM_HD_TIF_IMAGE_SAMPLES = {
    "Visium_HD_FF_Human_Breast_Cancer",
    "Visium_HD_FF_Human_Tonsil",
    "Visium_HD_Human_Breast_Cancer_Fresh_Frozen",
    "Visium_HD_Human_Kidney_FFPE",  # 3.1.2 only
    "Visium_HD_Human_Lung_Cancer",
    "Visium_HD_Human_Lymph_Node_FFPE",
    "Visium_HD_Human_Prostate_Cancer_FFPE",
    "Visium_HD_Human_Tonsil_Fresh_Frozen",
    "Visium_HD_Human_Tonsil_Fresh_Frozen_IF",
    "Visium_HD_Mouse_Brain",
    "Visium_HD_Mouse_Brain_Fresh_Frozen",
    "Visium_HD_Mouse_Lung_Fresh_Frozen",
}


def visium_sge(
    sample_id="V1_Breast_Cancer_Block_A_Section_1",
    *,
    spaceranger_version: str | None = None,
    base_url: str = VISIUM_BASE_URL,
    include_hires_tiff: bool = False,
    **reader_kwargs,
) -> spatialdata.SpatialData:
    """Processed Visium Spatial Gene Expression data from 10x Genomics' database.

    The database_ can be browsed online to find the ``sample_id`` you want.

    .. _database: https://support.10xgenomics.com/spatial-gene-expression/datasets

    Parameters
    ----------
    sample_id
        The ID of the data sample in 10x's spatial database.
    spaceranger_version: Version folder in the download URL. Defaults to
        1.1.0 for `V1_` samples, otherwise 1.2.0.
    base_url: Base URL for downloads, without the version folder.
    include_hires_tiff
        Download and include the high-resolution tissue image (tiff) in
        `adata.uns["spatial"][sample_id]["metadata"]["source_image_path"]`.
         Also download and load the full-resolution tissue image.

    Returns
    -------
    Spatial data object.
    """

    version = spaceranger_version or ("1.1.0" if "V1_" in sample_id else "1.2.0")
    url = f"{base_url}/{version}/{sample_id}/{sample_id}_"
    folder = settings.datasetdir / sample_id
    _retrieve(url + "filtered_feature_bc_matrix.h5", folder)
    _retrieve(url + "spatial.tar.gz", folder, members=None)
    if include_hires_tiff:
        reader_kwargs.setdefault(
            "fullres_image_file",
            _retrieve(url + "image.tif", folder),
        )
    return spatialdata_io.visium(folder, dataset_id=sample_id, **reader_kwargs)


def xenium_sge(
    sample_id: str = "Xenium_FFPE_Human_Breast_Cancer_Rep1",
    *,
    xenium_version: str | None = None,
    include_hires_tiff: bool = False,
    aligned_images: bool = True,
    base_url: str = XENIUM_BASE_URL,
    zip_filename: str = "outs.zip",
    image_filename: str = "he_image.ome.tif",
    alignment_filename: str = "he_imagealignment.csv",
    **reader_kwargs,
) -> spatialdata.SpatialData:
    """
     Xenium data from 10x Genomics' database.

    Args:
         sample_id: ID of the sample in 10x's database.
         xenium_version: Version of Xenium Onboard Analysis or Xenium Ranger that
             produced the outputs, the version folder in the download URL. Looked
             up in `XENIUM_SAMPLES` when not given.
         include_hires_tiff: Also download the aligned H&E image and its
             alignment matrix.
         aligned_images: Load the aligned H&E and IF images in the sample folder
             (`*he_image.ome.tif`, `*if_image.ome.tif`) as `sdata.images["he_image"]`
             and `sdata.images["if_image"]`; see `stlearn.io.xenium`.
         base_url: Base URL for downloads, without the version folder.
         zip_filename: Name of the Xenium output archive.
         image_filename: Name of the H&E image.
         alignment_filename: Name of the H&E image's alignment matrix.
         reader_kwargs: Passed to `spatialdata_io.xenium`, e.g. `transcripts=False`.

     Returns
     -------
     Spatial data object.
    """
    version = _version(sample_id, xenium_version, XENIUM_SAMPLES)
    url = f"{base_url}/{version}/{sample_id}/{sample_id}_"
    folder = settings.datasetdir / sample_id
    archive = _retrieve(url + zip_filename, folder, unpack=False)
    if not (folder / "experiment.xenium").exists():
        shutil.unpack_archive(archive, folder)
        _move_up(next(folder.rglob("experiment.xenium")).parent, folder)
    # Have image and alignment file to be without prefix of library id.
    if include_hires_tiff:
        _retrieve(url + image_filename, folder, fname=image_filename)
        # Aligned images are paired with "<name>alignment.csv"
        # (Atera: he_alignment.csv).
        stem = image_filename.removesuffix(".ome.tif")
        _retrieve(url + alignment_filename, folder, fname=f"{stem}alignment.csv")
    return xenium(folder, aligned_images=aligned_images, **reader_kwargs)


def atera_sge(
    sample_id: str = "WTA_Preview_FFPE_Breast_Cancer",
    *,
    atera_version: str | None = None,
    include_hires_tiff: bool = False,
    aligned_images: bool = True,
    base_url: str = ATERA_BASE_URL,
    **reader_kwargs,
) -> spatialdata.SpatialData:
    """Atera data from 10x Genomics' database.

    10x converted the preview datasets to Xenium Onboard Analysis v4 formats, so
    they are read with `spatialdata_io.xenium`.

    Args:
        sample_id: ID of the sample in 10x's database.
        atera_version: Version folder in the download URL. Looked up in
            `ATERA_SAMPLES` when not given.
        include_hires_tiff: Also download the aligned H&E image and its
            alignment matrix.
        aligned_images: Load the aligned H&E image, when downloaded, as
            `sdata.images["he_image"]`.
        base_url: Base URL for downloads, without the version folder.
        reader_kwargs: Passed to `spatialdata_io.xenium`, e.g. `transcripts=False`.

    Returns
    -------
    Spatial data object.
    """
    version = _version(sample_id, atera_version, ATERA_SAMPLES)
    return xenium_sge(
        sample_id,
        xenium_version=version,
        include_hires_tiff=include_hires_tiff,
        aligned_images=aligned_images,
        base_url=base_url,
        alignment_filename="he_alignment.csv",
        **reader_kwargs,
    )


def visium_hd_sge(
    sample_id: str = "Visium_HD_Human_Breast_Cancer_Fresh_Frozen",
    *,
    bin_size: int = 8,
    include_hires_tiff: bool = False,
    spaceranger_version: str | None = None,
    base_url: str = VISIUM_BASE_URL,
    image_filename: str | None = None,
    zarr_path: pathlib.Path | str | None = None,
    **reader_kwargs,
) -> spatialdata.SpatialData:
    """One bin size of a Visium HD dataset from 10x Genomics' database.

    Files are kept in `settings.datasetdir / sample_id` and downloaded only once.
    The binned outputs archive holds every bin size; only `bin_size` is unpacked.

    Will converts cell ranger formatted directories into a zarr store and will use that
    on subsequent calls.

    Args:
        sample_id: ID of the sample in 10x's database.
        bin_size: Bin size in µm (2, 8 or 16).
        include_hires_tiff: Also download and load the full-resolution tissue image.
        spaceranger_version: Version folder in the download URL. Looked up in
            `VISIUM_HD_SAMPLES` when not given.
        base_url: Base URL for downloads, without the version folder.
        image_filename: Name of the full-resolution tissue image.
        zarr_path: Location of zarr directory (used to store source and checkpoint data)
            Default is the `settings.datasetdir / sample_id /
            f"{sample_id}_square_{bin_size:03d}um.zarr"`.
        reader_kwargs: Passed to `spatialdata_io.visium_hd`.

    Returns
    -------
    Spatial data object.
    """
    version = _version(sample_id, spaceranger_version, VISIUM_HD_SAMPLES)
    url = f"{base_url}/{version}/{sample_id}/{sample_id}_"
    folder = settings.datasetdir / sample_id
    _retrieve(url + "feature_slice.h5", folder)
    _retrieve(url + "spatial.tar.gz", folder, members=None)
    _retrieve(
        url + "binned_outputs.tar.gz",
        folder,
        members=[f"binned_outputs/square_{bin_size:03d}um"],
    )
    if include_hires_tiff:
        if image_filename is None:
            tif = sample_id in VISIUM_HD_TIF_IMAGE_SAMPLES
            image_filename = "tissue_image.tif" if tif else "tissue_image.btf"
        _retrieve(url + image_filename, folder)
    if zarr_path is None:
        zarr_path = folder / f"{sample_id}_square_{bin_size:03d}um.zarr"
    return visium_hd(
        folder,
        dataset_id=sample_id,
        bin_size=bin_size,
        zarr_path=zarr_path,
        **reader_kwargs,
    )


def _version(sample_id: str, version: str | None, samples: dict[str, str]) -> str:
    """`version` if given, otherwise the sample's version from `samples`."""
    if version is not None:
        return version
    try:
        return samples[sample_id]
    except KeyError:
        raise ValueError(
            f"Unknown sample {sample_id!r}: pass its version, or "
            f"use one of {sorted(samples)}",
        ) from None


def _move_up(subfolder: pathlib.Path, folder: pathlib.Path) -> None:
    if subfolder == folder:
        return
    for item in subfolder.iterdir():
        target = folder / item.name
        if target.is_dir():
            shutil.rmtree(target)
        item.replace(target)
    subfolder.rmdir()


def _retrieve(
    url: str,
    folder: pathlib.Path,
    members: list[str] | None = None,
    fname: str | None = None,
    unpack: bool = True,
) -> str:
    """Download `url` into `folder` unless it's there, unpacking archives there.

    `members` is None to unpack the whole archive, or a list of the files and
    folders in it to unpack. Archives are kept, so later calls skip the download.
    `fname` is the name to save under; defaults to the URL's file name.
    `unpack=False` keeps an archive packed.
    """
    processor = None
    if unpack and url.endswith(".zip"):
        processor = pooch.Unzip(members=members, extract_dir=".")
    elif unpack and url.endswith(".tar.gz"):
        processor = pooch.Untar(members=members, extract_dir=".")

    kwargs = dict(
        known_hash=None,
        fname=fname or url.rsplit("/", 1)[-1],
        path=folder,
        processor=processor,
        progressbar=True,
    )
    # Try direct and fall back to s3.
    try:
        return pooch.retrieve(url, **kwargs)
    except requests.HTTPError as err:
        too_large = err.response is not None and err.response.status_code == 400
        if not (too_large and url.startswith(_CDN_URL)):
            raise
        return pooch.retrieve(_S3_URL + url.removeprefix(_CDN_URL), **kwargs)
