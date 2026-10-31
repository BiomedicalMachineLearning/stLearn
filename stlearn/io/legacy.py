"""Reading and Writing"""

import collections.abc
import os
import pathlib

import anndata
import numpy
import pandas
import PIL.Image as PIL_Image
import scanpy

import stlearn
from stlearn.types import _BACKGROUND


def read_old_st(
    count_matrix_file: os.PathLike[str] | str | collections.abc.Iterator[str],
    spatial_file: int | str | bytes | os.PathLike[str] | os.PathLike[bytes],
    image_file: str | pathlib.Path,
    library_id: str = "OldST",
    scale: float = 1.0,
    quality: str = "hires",
    spot_diameter_fullres: float = 50,
) -> anndata.AnnData:
    """\
    Read Old Spatial Transcriptomics data. Must set an

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


def read_slide_seq(
    count_matrix_file: str | pathlib.Path,
    spatial_file: str | pathlib.Path,
    library_id: str | None = None,
    scale: float | None = None,
    quality: str = "hires",
    spot_diameter_fullres: float = 50,
    background_color: _BACKGROUND = "white",
) -> anndata.AnnData:
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

    count = pandas.read_csv(count_matrix_file)
    meta = pandas.read_csv(spatial_file)

    adata = anndata.AnnData(count.iloc[:, 1:].set_index("gene").T)

    adata.var["ENSEMBL"] = count["ENSEMBL"].values

    adata.obs["index"] = meta["index"].values

    if scale is None:
        max_coor = numpy.max(meta[["x", "y"]].values)
        scale = 2000 / max_coor

    adata.obs["imagecol"] = meta["x"].values * scale
    adata.obs["imagerow"] = meta["y"].values * scale

    # Create image
    max_size = numpy.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = PIL_Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = PIL_Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = numpy.array(image)

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


def read_merfish(
    count_matrix_file: str | pathlib.Path,
    spatial_file: str | pathlib.Path,
    library_id: str | None = None,
    scale: float | None = None,
    quality: str = "hires",
    spot_diameter_fullres: float = 10,
    background_color: _BACKGROUND = "white",
    *,
    sheet_name: str | int | list[str | int] | None = 0,
    blank_prefix: str | None = "Blank-",
    ncols: int | None = None,
    tile_size: int = 1000,
) -> anndata.AnnData:
    """\
    Read MERFISH data

    Reads a count matrix (CSV, genes x cells) and the cell positions in microns
    (an Excel workbook with one sheet per replicate), the layout of the
    supplementary data of Xia et al. (2019), PNAS 116(39):19490.

    Positions stay in microns, so one full-resolution pixel is one micron, and
    `scalefactors["microns_per_pixel"]` is 1.0. When a sheet has negative
    positions, all of its positions are moved by the most negative value plus 100.

    Barcodes whose names start with `blank_prefix` are blank controls: barcodes
    assigned to no gene, whose counts are decoding errors. They are left out of
    `X` and kept in `obsm["blank"]`, with each cell's blank total in
    `obs["blank_counts"]` and its share of all the cell's barcode counts in
    `obs["blank_fraction"]`.

    Parameters
    ----------
    count_matrix_file
        Path to the count matrix: genes in rows, cells in columns, gene names in
        the first column and cell IDs in the header.
    spatial_file
        Path to the Excel workbook of positions: cell IDs in the first column,
        then x and y in microns.
    library_id
        Key under `adata.uns["spatial"]`. Defaults to "MERSEQ".
    scale
        Image pixels per micron. Defaults to fitting each replicate in an image
        about 2,000 pixels across.
    quality
        Set quality that convert to stlearn to use. Store in
        anndata.obs['imagecol' & 'imagerow']
    spot_diameter_fullres
        Typical cell diameter, in microns (full-resolution pixels). stLearn uses it
        where it needs a spot size: `st.tl.cci.run` with `distance=None` looks for
        neighbours within twice this.
    background_color
        Color of the background. Only `black` or `white` is allowed.
    sheet_name
        Sheets of `spatial_file` to read, as in `pandas.read_excel`: a sheet name
        or position for one replicate, a list for several, or None for all of
        them. Several replicates are laid out side by side with
        `stlearn.concatenate_spatial_adata`, with each cell's sheet name in
        `obs["batch"]`.
    blank_prefix
        Name prefix of the blank control barcodes. None reads every row as a gene.
    ncols
        Number of columns to lay several replicates out in. Defaults to one row.
    tile_size
        Width and height, in pixels, of each replicate's tile when laying out
        several.

    Returns
    -------
    AnnData
    """

    positions = pandas.read_excel(spatial_file, sheet_name=sheet_name, index_col=0)
    # Genes by cells, with gene names in the first column and cell IDs as the header.
    counts = pandas.read_csv(count_matrix_file, index_col=0)
    counts.index = counts.index.astype(str)
    counts.columns = counts.columns.astype(str)
    if blank_prefix is None:
        is_blank = numpy.zeros(len(counts), dtype=bool)
    else:
        is_blank = numpy.asarray(counts.index.str.startswith(blank_prefix), dtype=bool)

    def replicate(sheet: pandas.DataFrame, name: str) -> anndata.AnnData:
        return _merfish_library(
            counts,
            is_blank,
            sheet,
            name,
            scale,
            quality,
            spot_diameter_fullres,
            background_color,
        )

    library_id = library_id or "MERSEQ"
    if isinstance(positions, pandas.DataFrame):
        return replicate(positions, library_id)

    names = [str(name) for name in positions]
    replicates = [
        replicate(sheet, name)
        for name, sheet in zip(names, positions.values(), strict=True)
    ]
    if len(replicates) == 1:
        adata = replicates[0]
    else:
        adata = stlearn.concatenate_spatial_adata(
            replicates,
            ncols=ncols or len(replicates),
            fixed_size=(tile_size, tile_size),
        )
        adata.obs["batch"] = adata.obs["batch"].cat.rename_categories(names)
        # concatenate makes the cell IDs unique with a suffix; keep the IDs from
        # the files when they already are.
        cell_ids = numpy.concatenate([r.obs_names.to_numpy() for r in replicates])
        if len(set(cell_ids)) == len(cell_ids):
            adata.obs_names = cell_ids
    # The library id is the first replicate's; use the requested one.
    adata.uns["spatial"] = {library_id: adata.uns["spatial"].pop(names[0])}
    return adata


def _merfish_library(
    counts: pandas.DataFrame,
    is_blank: numpy.ndarray,
    coordinates: pandas.DataFrame,
    library_id: str,
    scale: float | None,
    quality: str,
    spot_diameter_fullres: float,
    background_color: _BACKGROUND,
) -> anndata.AnnData:
    """One replicate of `read_merfish`: the cells in `coordinates`, with a blank
    image and scale factors under `uns["spatial"][library_id]`."""
    coordinates = coordinates.iloc[:, :2].copy()
    # turn ids into strings (obs_names).
    coordinates.index = coordinates.index.astype(str)
    missing = coordinates.index.difference(counts.columns)
    if len(missing) > 0:
        raise ValueError(
            f"{len(missing)} cells with positions are not in the count matrix, "
            f"e.g. {list(missing[:3])}."
        )
    if coordinates.min().min() < 0:
        coordinates = coordinates + numpy.abs(coordinates.min().min()) + 100

    cell_counts = counts[coordinates.index]
    adata_merfish = anndata.AnnData(cell_counts.loc[~is_blank].T.astype(numpy.float32))
    if is_blank.any():
        blank = cell_counts.loc[is_blank].T.astype(numpy.float32)
        blank_counts = blank.sum(axis=1).to_numpy()
        total_counts = cell_counts.sum(axis=0).to_numpy(dtype=numpy.float32)
        adata_merfish.obsm["blank"] = blank
        adata_merfish.obs["blank_counts"] = blank_counts
        adata_merfish.obs["blank_fraction"] = numpy.divide(
            blank_counts,
            total_counts,
            out=numpy.zeros_like(total_counts),
            where=total_counts > 0,
        )
    adata_merfish.obsm["spatial"] = coordinates.to_numpy()

    if scale is None:
        max_coor = numpy.max(adata_merfish.obsm["spatial"])
        scale = 2000 / max_coor

    adata_merfish.obs["imagecol"] = adata_merfish.obsm["spatial"][:, 0] * scale
    adata_merfish.obs["imagerow"] = adata_merfish.obsm["spatial"][:, 1] * scale

    # Create image
    max_size = numpy.max(
        [adata_merfish.obs["imagecol"].max(), adata_merfish.obs["imagerow"].max()],
    )
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = PIL_Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = PIL_Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
    imgarr = numpy.array(image)

    adata_merfish.uns["spatial"] = {}
    adata_merfish.uns["spatial"][library_id] = {}
    adata_merfish.uns["spatial"][library_id]["images"] = {}
    adata_merfish.uns["spatial"][library_id]["images"][quality] = imgarr
    adata_merfish.uns["spatial"][library_id]["use_quality"] = quality
    adata_merfish.uns["spatial"][library_id]["scalefactors"] = {
        "tissue_" + quality + "_scalef": scale,
        "spot_diameter_fullres": spot_diameter_fullres,
        # Positions are in microns, so one full-resolution pixel is one micron.
        "microns_per_pixel": 1.0,
    }

    return adata_merfish


def read_seq_fish(
    count_matrix_file: str | pathlib.Path,
    spatial_file: str | pathlib.Path,
    library_id: str | None = None,
    scale: float | None = 1.0,
    quality: str = "hires",
    field: int = 0,
    spot_diameter_fullres: float = 50,
    background_color: _BACKGROUND = "white",
) -> anndata.AnnData:
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

    count = pandas.read_table(count_matrix_file, header=None)
    spatial = pandas.read_table(spatial_file, index_col=False)

    count = count.T
    count.columns = count.iloc[0].to_numpy()
    count = count.drop(count.index[0]).reset_index(drop=True)
    count = count[count["Field_of_View"] == field].drop(count.columns[[0, 1]], axis=1)

    spatial = spatial[spatial["Field_of_View"] == field]

    adata = anndata.AnnData(count.astype(numpy.float32))

    if scale is None:
        max_coor = numpy.max(spatial[["X", "Y"]])
        scale = 2000 / max_coor

    adata.obs["imagecol"] = spatial["X"].values * scale
    adata.obs["imagerow"] = spatial["Y"].values * scale

    adata.obsm["spatial"] = spatial[["X", "Y"]].values

    # Create image
    max_size = numpy.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = PIL_Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = PIL_Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = numpy.array(image)

    if library_id is None:
        library_id = "SeqFISH"

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
