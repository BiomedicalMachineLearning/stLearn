"""Reading and Writing"""

import json
import logging
import pathlib

import anndata
import h5py
import pandas
import scanpy

from stlearn.types import _QUALITY

from ..utils import _quality_scalef, read_image
from .spatialdata import (
    attach_spatial_sdata,
    visium_scalefactors,
    xenium,
    xenium_pixel_size,
)


def read_10x(
    path: str | pathlib.Path,
    genome: str | None = None,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str | None = None,
    load_images: bool = True,
    quality: _QUALITY = "hires",
    image_path: str | pathlib.Path | None = None,
) -> anndata.AnnData:
    """\
    Read data from 10X.

    In addition to reading regular 10x output, this looks for the `spatial` folder
    and loads images, coordinates and scale factors. Based on the
    https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview

    Parameters
    ----------
    path:
        The path to directory for the datafiles.
    genome:
        Filter expression to genes within this genome.
    count_file:
        Which file in the directory to use as the count file. Typically, it would be one
        of: 'filtered_feature_bc_matrix.h5' or 'raw_feature_bc_matrix.h5'.
    library_id:
        Identifier for the library. Can be modified when concatenating multiple
        adata objects.
    load_images:
        Load image or not.
    quality:
        Set quality that convert to stlearn to use. Store in
        anndata.obs['imagecol' & 'imagerow']
    image_path:
        Path to image. Only need when loading full resolution image.


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
        Dict of images (`'fulres'`, `'hires'` and `'lowres'`)
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['scalefactors']`
        Scale factors for the spots
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]['metadata']`
        Files metadata: 'chemistry_description', 'software_version'
    :attr:`~anndata.AnnData.obsm`\\ `['spatial']`
        Spatial spot coordinates, usable as `basis` by :func:`~scanpy.pl.embedding`.
    """

    path = pathlib.Path(path)
    adata = scanpy.read_10x_h5(path / count_file, genome=genome)

    adata.uns["spatial"] = dict()

    with h5py.File(path / count_file, mode="r") as f:
        attrs = dict(f.attrs)

    if library_id is None:
        library_id = str(attrs.pop("library_ids")[0], "utf-8")
    adata.uns["spatial"][library_id] = dict()

    tissue_positions_file = (
        path / "spatial/tissue_positions.csv"
        if (path / "spatial/tissue_positions.csv").exists()
        else path / "spatial/tissue_positions_list.csv"
    )

    if load_images:
        files = dict(
            tissue_positions_file=tissue_positions_file,
            scalefactors_json_file=path / "spatial/scalefactors_json.json",
            hires_image=path / "spatial/tissue_hires_image.png",
            lowres_image=path / "spatial/tissue_lowres_image.png",
        )

        adata.uns["spatial"][library_id]["images"] = dict()
        for res in ["hires", "lowres"]:
            image_file = files[f"{res}_image"]
            if not image_file.exists():
                raise OSError(f"Could not find '{image_file}'")
            try:
                adata.uns["spatial"][library_id]["images"][res] = read_image(image_file)
            except Exception as err:
                raise OSError(f"Could not read '{image_file}': {err}") from err

        # read json scalefactors
        adata.uns["spatial"][library_id]["scalefactors"] = visium_scalefactors(
            json.loads(files["scalefactors_json_file"].read_bytes())
        )

        adata.uns["spatial"][library_id]["metadata"] = {
            k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
            for k in ("chemistry_description", "software_version")
            if k in attrs
        }

        # read coordinates - tissue_positions.csv has a header row.
        header = 0 if tissue_positions_file.name == "tissue_positions.csv" else None
        positions = pandas.read_csv(tissue_positions_file, header=header)

        positions.columns = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ]
        positions.index = positions["barcode"]

        adata.obs = adata.obs.join(positions, how="left")

        adata.obsm["spatial"] = (
            adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]]
            .to_numpy()
            .astype(float)
        )
        for col in ["barcode", "pxl_row_in_fullres", "pxl_col_in_fullres"]:
            if col in adata.obs.columns:
                del adata.obs[col]

        if quality == "fulres":
            # put image path in uns
            if image_path is not None:
                # get an absolute path
                image_path = str(pathlib.Path(image_path).resolve())
                adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                    image_path,
                )
            else:
                raise ValueError("Trying to load fulres but no image_path set.")

            image_coor = adata.obsm["spatial"]
            img = read_image(image_path)
            adata.uns["spatial"][library_id]["images"]["fulres"] = img
        else:
            scale = _quality_scalef(adata.uns["spatial"][library_id], quality)
            image_coor = adata.obsm["spatial"] * scale

        adata.obs["imagecol"] = image_coor[:, 0]
        adata.obs["imagerow"] = image_coor[:, 1]
        adata.uns["spatial"][library_id]["use_quality"] = quality

        adata.obs["array_row"] = adata.obs["array_row"].astype(int)
        adata.obs["array_col"] = adata.obs["array_col"].astype(int)
        adata.obsm["spatial"] = adata.obsm["spatial"].astype("int64")

    adata.var_names_make_unique()

    return adata


# Arguments of the file-based read_xenium (stLearn <= 1.4), now read from the folder.
_FILE_BASED_ARGS = {
    "feature_cell_matrix_file",
    "cell_summary_file",
    "image_path",
    "alignment_matrix_file",
    "experiment_xenium_file",
    "default_pixel_size_microns",
    "scale",
    "quality",
    "spot_diameter_fullres",
    "background_color",
}

# Elements of the output folder the table doesn't need.
_SKIPPED_ELEMENTS = (
    "cells_boundaries",
    "nucleus_boundaries",
    "nucleus_labels",
    "transcripts",
    "morphology_mip",
    "morphology_focus",
)


def read_xenium(
    path: str | pathlib.Path,
    library_id: str | None = None,
    image_key: str | None = "he_image",
    use_quality: str = "hires",
    max_px: int | None = 4000,
    spot_diameter_um: float = 10.0,
    **reader_kwargs,
) -> anndata.AnnData:
    """\
    Read a Xenium or Atera output folder.

    Returns the cells' table ready for stLearn's tools and plots: the folder is read
    with `stlearn.io.xenium`, which aligns the H&E to the cells, then
    `attach_spatial_sdata` adds the image and scale factors. To keep the SpatialData
    object (other elements, cropping, checkpoints), call those two yourself.

    Coordinates are in pixels of the full-resolution aligned image.

    Parameters
    ----------
    path
        The output folder, with `experiment.xenium` and, for the H&E,
        `he_image.ome.tif` and `he_imagealignment.csv`.
    library_id
        Key under `adata.uns["spatial"]`. Defaults to the folder name.
    image_key
        Aligned image to show behind the cells: "he_image", "if_image", or None
        for a blank image. Falls back to blank, with a warning, when the folder
        doesn't have it.
    use_quality
        Name of the image resolution, "hires" or "lowres".
    max_px
        Longest side of the image, in pixels.
    spot_diameter_um
        Typical cell diameter, where stLearn needs a spot size.
    reader_kwargs
        Passed to `spatialdata_io.xenium`. Elements the table doesn't need
        (boundaries, transcripts, morphology images, nucleus labels) are skipped
        unless set here.

    Returns
    -------
    Annotated data matrix of cells by genes, with:
    :attr:`~anndata.AnnData.obsm`\\ `['spatial']`
        Cell centroids in full-resolution pixels.
    :attr:`~anndata.AnnData.obs`\\ `['imagecol', 'imagerow']`
        Cell centroids in pixels of `images[use_quality]`.
    :attr:`~anndata.AnnData.uns`\\ `['spatial'][library_id]`
        `images`, `use_quality` and `scalefactors`, with `microns_per_pixel` the
        microns per full-resolution pixel.
    """
    path = pathlib.Path(path)
    file_based = sorted(_FILE_BASED_ARGS & reader_kwargs.keys())
    if file_based or path.is_file():
        raise TypeError(
            "read_xenium now reads the Xenium/Atera output folder, e.g. "
            "read_xenium(data_dir), and finds the matrix, cells, experiment.xenium "
            "and H&E alignment there. Coordinates are full-resolution pixels of the "
            "aligned H&E in obsm['spatial'], as before, and pixels of the downsampled "
            "image in obs['imagecol'/'imagerow']."
            + (f" No longer taken: {', '.join(file_based)}." if file_based else ""),
        )
    for element in _SKIPPED_ELEMENTS:
        reader_kwargs.setdefault(element, False)

    sdata = xenium(path, aligned_images=image_key is not None, **reader_kwargs)
    if image_key is not None and image_key not in sdata.images:
        logging.warning(f"No {image_key!r} in {path}; using a blank image.")
        image_key = None
    return attach_spatial_sdata(
        sdata.tables["table"],
        sdata,
        library_id or path.name,
        xenium_pixel_size(path),
        image_key=image_key,
        use_quality=use_quality,
        max_px=max_px,
        spot_diameter_um=spot_diameter_um,
    )
