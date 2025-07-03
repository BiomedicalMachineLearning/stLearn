"""Reading and Writing"""

import json
import logging as logg
from collections.abc import Iterator
from os import PathLike
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
from anndata import AnnData
from matplotlib.image import imread
from PIL import Image

import stlearn
from stlearn.types import _BACKGROUND, _QUALITY
from stlearn.wrapper.xenium_alignment import apply_alignment_transformation


def Read10X(
    path: str | Path,
    genome: str | None = None,
    count_file: str = "filtered_feature_bc_matrix.h5",
    library_id: str | None = None,
    load_images: bool = True,
    quality: _QUALITY = "hires",
    image_path: str | Path | None = None,
) -> AnnData:
    """\
    Read data from 10X.

    In addition to reading regular 10x output, this looks for the `spatial` folder
    and loads images, coordinates and scale factors. Based on the
    `Space Ranger output docs`_.

    _Space Ranger output docs:
    https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview

    Parameters
    ----------
    path
        The path to directory for the datafiles.
    genome
        Filter expression to genes within this genome.
    count_file
        Which file in the directory to use as the count file. Typically, it would be one
        of: 'filtered_feature_bc_matrix.h5' or 'raw_feature_bc_matrix.h5'.
    library_id
        Identifier for the library. Can be modified when concatenating multiple
        adata objects.
    load_images
        Load image or not.
    quality
        Set quality that convert to stlearn to use. Store in
        anndata.obs['imagecol' & 'imagerow']
    image_path
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

    path = Path(path)
    adata = scanpy.read_10x_h5(path / count_file, genome=genome)

    adata.uns["spatial"] = dict()

    from h5py import File

    with File(path / count_file, mode="r") as f:
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

        # check if files exists, continue if images are missing
        for f in files.values():
            if not f.exists():
                if any(x in str(f) for x in ["hires_image", "lowres_image"]):
                    logg.warning(
                        f"You seem to be missing an image file.\nCould not find '{f}'."
                    )
                else:
                    raise OSError(f"Could not find '{f}'")

        adata.uns["spatial"][library_id]["images"] = dict()
        for res in ["hires", "lowres"]:
            try:
                adata.uns["spatial"][library_id]["images"][res] = imread(
                    str(files[f"{res}_image"])
                )
            except Exception:
                raise OSError(f"Could not find '{res}_image'")

        # read json scalefactors
        adata.uns["spatial"][library_id]["scalefactors"] = json.loads(
            files["scalefactors_json_file"].read_bytes()
        )

        adata.uns["spatial"][library_id]["metadata"] = {
            k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
            for k in ("chemistry_description", "software_version")
            if k in attrs
        }

        # read coordinates
        positions = pd.read_csv(files["tissue_positions_file"], header=None)
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
        adata.obs.drop(
            columns=["barcode", "pxl_row_in_fullres", "pxl_col_in_fullres"],
            inplace=True,
        )

        if quality == "fulres":
            # put image path in uns
            if image_path is not None:
                # get an absolute path
                image_path = str(Path(image_path).resolve())
                adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                    image_path
                )
            else:
                raise ValueError("Trying to load fulres but no image_path set.")

            image_coor = adata.obsm["spatial"]
            img = plt.imread(image_path, None)
            adata.uns["spatial"][library_id]["images"]["fulres"] = img
        else:
            scale = adata.uns["spatial"][library_id]["scalefactors"][
                "tissue_" + quality + "_scalef"
            ]
            image_coor = adata.obsm["spatial"] * scale

        adata.obs["imagecol"] = image_coor[:, 0]
        adata.obs["imagerow"] = image_coor[:, 1]
        adata.uns["spatial"][library_id]["use_quality"] = quality

    adata.var_names_make_unique()

    adata.obs["array_row"] = adata.obs["array_row"].astype(int)
    adata.obs["array_col"] = adata.obs["array_col"].astype(int)
    adata.obsm["spatial"] = adata.obsm["spatial"].astype("int64")

    return adata


def ReadOldST(
    count_matrix_file: PathLike[str] | str | Iterator[str],
    spatial_file: int | str | bytes | PathLike[str] | PathLike[bytes],
    image_file: str | Path | None = None,
    library_id: str = "OldST",
    scale: float = 1.0,
    quality: str = "hires",
    spot_diameter_fullres: float = 50,
) -> AnnData:
    """\
    Read Old Spatial Transcriptomics data

    Parameters
    ----------
    count_matrix_file
        Path to count matrix file.
    spatial_file
        Path to the spatial location file.
    image_file
        Path to the tissue image file
    library_id
        Identifier for the library. Can be modified when concatenating multiple
        adata objects.
    scale
        Set scale factor.
    quality
        Set quality that convert to stlearn to use. Store in
        anndata.obs['imagecol' & 'imagerow']
    spot_diameter_fullres
        Diameter of spot in full resolution

    Returns
    -------
    AnnData
    """

    adata = scanpy.read_text(count_matrix_file)
    adata = stlearn.add.parsing(adata, coordinates_file=spatial_file)
    stlearn.add.image(
        adata,
        library_id=library_id,
        quality=quality,
        imgpath=image_file,
        scale=scale,
        spot_diameter_fullres=spot_diameter_fullres,
    )

    return adata


def ReadSlideSeq(
    count_matrix_file: str | Path,
    spatial_file: str | Path,
    library_id: str | None = None,
    scale: float | None = None,
    quality: str = "hires",
    spot_diameter_fullres: float = 50,
    background_color: _BACKGROUND = "white",
) -> AnnData:
    """\
    Read Slide-seq data

    Parameters
    ----------
    count_matrix_file
        Path to count matrix file.
    spatial_file
        Path to the spatial location file.
    library_id
        Identifier for the library. Can be modified when concatenating
        multiple adata objects.
    scale
        Set scale factor.
    quality
        Set quality that convert to stlearn to use. Store in
        anndata.obs['imagecol' & 'imagerow']
    spot_diameter_fullres
        Diameter of spot in full resolution
    background_color
        Color of the background. Only `black` or `white` is allowed.

    Returns
    -------
    AnnData
    """

    count = pd.read_csv(count_matrix_file)
    meta = pd.read_csv(spatial_file)

    adata = AnnData(count.iloc[:, 1:].set_index("gene").T)

    adata.var["ENSEMBL"] = count["ENSEMBL"].values

    adata.obs["index"] = meta["index"].values

    if scale is None:
        max_coor = np.max(meta[["x", "y"]].values)
        scale = 2000 / max_coor

    adata.obs["imagecol"] = meta["x"].values * scale
    adata.obs["imagerow"] = meta["y"].values * scale

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "Slide-seq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"][
        "tissue_" + quality + "_scalef"
    ] = scale

    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"
    ] = spot_diameter_fullres
    adata.obsm["spatial"] = meta[["x", "y"]].values

    return adata


def ReadMERFISH(
    count_matrix_file: str | Path,
    spatial_file: str | Path,
    library_id: str | None = None,
    scale: float | None = None,
    quality: str = "hires",
    spot_diameter_fullres: float = 50,
    background_color: _BACKGROUND = "white",
) -> AnnData:
    """\
    Read MERFISH data

    Parameters
    ----------
    count_matrix_file
        Path to count matrix file.
    spatial_file
        Path to the spatial location file.
    library_id
        Identifier for the library. Can be modified when concatenating
        multiple adata objects.
    scale
        Set scale factor.
    quality
        Set quality that convert to stlearn to use. Store in
        anndata.obs['imagecol' & 'imagerow']
    spot_diameter_fullres
        Diameter of spot in full resolution
    background_color
        Color of the background. Only `black` or `white` is allowed.

    Returns
    -------
    AnnData
    """

    coordinates = pd.read_excel(spatial_file, index_col=0)
    if coordinates.min().min() < 0:
        coordinates = coordinates + np.abs(coordinates.min().min()) + 100
    from scanpy import read_csv

    counts = read_csv(count_matrix_file).transpose()

    adata_merfish = counts[coordinates.index, :]
    adata_merfish.obsm["spatial"] = coordinates.to_numpy()

    if scale is None:
        max_coor = np.max(adata_merfish.obsm["spatial"])
        scale = 2000 / max_coor

    adata_merfish.obs["imagecol"] = adata_merfish.obsm["spatial"][:, 0] * scale
    adata_merfish.obs["imagerow"] = adata_merfish.obsm["spatial"][:, 1] * scale

    # Create image
    max_size = np.max(
        [adata_merfish.obs["imagecol"].max(), adata_merfish.obs["imagerow"].max()]
    )
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "MERSEQ"

    adata_merfish.uns["spatial"] = {}
    adata_merfish.uns["spatial"][library_id] = {}
    adata_merfish.uns["spatial"][library_id]["images"] = {}
    adata_merfish.uns["spatial"][library_id]["images"][quality] = imgarr
    adata_merfish.uns["spatial"][library_id]["use_quality"] = quality
    adata_merfish.uns["spatial"][library_id]["scalefactors"] = {}
    adata_merfish.uns["spatial"][library_id]["scalefactors"][
        "tissue_" + quality + "_scalef"
    ] = scale
    adata_merfish.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"
    ] = spot_diameter_fullres

    return adata_merfish


def ReadSeqFish(
    count_matrix_file: str | Path,
    spatial_file: str | Path,
    library_id: str | None = None,
    scale: float = 1.0,
    quality: str = "hires",
    field: int = 0,
    spot_diameter_fullres: float = 50,
    background_color: _BACKGROUND = "white",
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
        Identifier for the library. Can be modified when concatenating multiple
        adata objects.
    scale
        Set scale factor.
    quality
        Set quality that convert to stlearn to use. Store in
        anndata.obs['imagecol' & 'imagerow']
    field
        Set field of view for SeqFish data
    spot_diameter_fullres
        Diameter of spot in full resolution
    background_color
        Color of the background. Only `black` or `white` is allowed.
    Returns
    -------
    AnnData
    """

    count = pd.read_table(count_matrix_file, header=None)
    spatial = pd.read_table(spatial_file, index_col=False)

    count = count.T
    count.columns = count.iloc[0]
    count = count.drop(count.index[0]).reset_index(drop=True)
    count = count[count["Field_of_View"] == field].drop(count.columns[[0, 1]], axis=1)

    spatial = spatial[spatial["Field_of_View"] == field]

    adata = AnnData(count)

    if scale is None:
        max_coor = np.max(spatial[["X", "Y"]])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = spatial["X"].values * scale
    adata.obs["imagerow"] = spatial["Y"].values * scale

    adata.obsm["spatial"] = spatial[["X", "Y"]].values

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "Slide-seq"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"][
        "tissue_" + quality + "_scalef"
    ] = scale
    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"
    ] = spot_diameter_fullres

    return adata


def ReadXenium(
    feature_cell_matrix_file: str | Path,
    cell_summary_file: str | Path,
    image_path: Path | None = None,
    library_id: str | None = None,
    scale: float = 1.0,
    quality: str = "hires",
    spot_diameter_fullres: float = 15,
    background_color: _BACKGROUND = "white",
    alignment_matrix_file: str | Path | None = None,
    experiment_xenium_file: str | Path | None = None,
    default_pixel_size_microns: float = 0.2125,
) -> AnnData:
    """\
    Read Xenium data

    Parameters
    ----------
    feature_cell_matrix_file
        Path to feature cell count matrix file. Only support h5ad file now.
    cell_summary_file
        Path to cell summary CSV file.
    image_path
        Path to image. Only need when loading full resolution image.
    library_id
        Identifier for the Xenium library. Can be modified when concatenating multiple
        adata objects.
    scale
        Set scale factor.
    quality
        Set quality that convert to stlearn to use. Store in
        anndata.obs['imagecol' & 'imagerow']
    spot_diameter_fullres
        Diameter of spot in full resolution
    background_color
        Color of the background. Only `black` or `white` is allowed.
    alignment_matrix_file
        Path to transformation matrix CSV file exported from Xenium Explorer.
        If provided, coordinates will be transformed according to coordinate_space.
    experiment_xenium_file
        Path to experiment.xenium JSON file. If provided, pixel_size will be read from
        here.
    default_pixel_size_microns
        Pixel size in microns (default 0.2125 for Xenium data).
    Returns
    -------
    AnnData
    """

    metadata = pd.read_csv(cell_summary_file)

    adata = scanpy.read_10x_h5(feature_cell_matrix_file)

    # Get original spatial coordinates
    spatial = metadata[["x_centroid", "y_centroid"]].copy()

    # Get pixel size from experiment.xenium file or use parameter
    if experiment_xenium_file is not None:
        with open(experiment_xenium_file) as f:
            experiment_data = json.load(f)
        pixel_size_microns = experiment_data.get("pixel_size")
    else:
        pixel_size_microns = default_pixel_size_microns
        print(
            f"Warning: Using default pixel size of {pixel_size_microns} microns. "
            "Consider providing experiment_xenium_file for accurate pixel size."
        )

    # Get and apply alignment transformation if provided
    if alignment_matrix_file is not None:
        transform_mat = pd.read_csv(alignment_matrix_file, header=None).values
        spatial = apply_alignment_transformation(
            spatial,
            transform_mat,
            pixel_size_microns,
        )

    spatial.columns = ["imagecol", "imagerow"]
    adata.obsm["spatial"] = spatial.values

    if scale is None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 2000 / max_coor

    if library_id is None:
        library_id = "Xenium_data"

    adata.obs["imagecol"] = spatial["imagecol"].values * scale
    adata.obs["imagerow"] = spatial["imagerow"].values * scale

    if image_path is not None:
        stlearn.add.image(
            adata,
            library_id=library_id,
            quality=quality,
            imgpath=image_path,
            scale=scale,
        )
    else:
        # Create image
        max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
        max_size = int(max_size + 0.1 * max_size)

        if background_color == "black":
            image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
        else:
            image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
        imgarr = np.array(image)

        # Create spatial dictionary
        adata.uns["spatial"] = {}
        adata.uns["spatial"][library_id] = {}
        adata.uns["spatial"][library_id]["images"] = {}
        adata.uns["spatial"][library_id]["images"][quality] = imgarr
        adata.uns["spatial"][library_id]["use_quality"] = quality
        adata.uns["spatial"][library_id]["scalefactors"] = {}
        adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"
        ] = scale
        adata.uns["spatial"][library_id]["scalefactors"][
            "spot_diameter_fullres"
        ] = spot_diameter_fullres

    return adata


def create_stlearn(
    count: pd.DataFrame,
    spatial: pd.DataFrame,
    library_id: str,
    image_path: Path | None = None,
    scale: float | None = None,
    quality: str = "hires",
    spot_diameter_fullres: float = 50,
    background_color: _BACKGROUND = "white",
):
    """\
    Create AnnData object for stLearn

    Parameters
    ----------
    count
        Pandas Dataframe of count matrix with rows as barcodes and columns as gene names
    spatial
        Pandas Dataframe of spatial location of cells/spots.
    library_id
        Identifier for the library. Can be modified when concatenating multiple
        adata objects.
    scale
        Set scale factor.
    quality
        Set quality that convert to stlearn to use. Store in
        anndata.obs['imagecol' & 'imagerow']
    spot_diameter_fullres
        Diameter of spot in full resolution
    background_color
        Color of the background. Only `black` or `white` is allowed.
    Returns
    -------
    AnnData
    """
    adata = AnnData(X=count)

    adata.obsm["spatial"] = spatial.values

    if scale is None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = spatial["imagecol"].values * scale
    adata.obs["imagerow"] = spatial["imagerow"].values * scale

    if image_path is not None:
        stlearn.add.image(
            adata,
            library_id=library_id,
            quality=quality,
            imgpath=image_path,
            scale=scale,
        )
    else:
        # Create image
        max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
        max_size = int(max_size + 0.1 * max_size)

        if background_color == "black":
            image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
        else:
            image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
        imgarr = np.array(image)

        # Create spatial dictionary
        adata.uns["spatial"] = {}
        adata.uns["spatial"][library_id] = {}
        adata.uns["spatial"][library_id]["images"] = {}
        adata.uns["spatial"][library_id]["images"][quality] = imgarr
        adata.uns["spatial"][library_id]["use_quality"] = quality
        adata.uns["spatial"][library_id]["scalefactors"] = {}
        adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"
        ] = scale
        adata.uns["spatial"][library_id]["scalefactors"][
            "spot_diameter_fullres"
        ] = spot_diameter_fullres

    return adata
