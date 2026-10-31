import json
import pathlib
from typing import Any, Literal

import anndata
import dask.array as da
import numpy
import pandas as pd
import spatialdata
import spatialdata_io
import tifffile
import xarray as xr
from skimage.transform import resize
from spatialdata.models import Image2DModel
from spatialdata.transformations import Affine, Identity, get_transformation

from ..utils import as_uint8_image, read_image
from ..wrapper.convert_scanpy import convert_scanpy

# Every Visium spot is 55 µm across.
VISIUM_SPOT_DIAMETER_UM = 55.0


def visium(
    path,
    *,
    dataset_id=None,
    zarr_path=None,
    **reader_kwargs,
) -> spatialdata.SpatialData:
    return _read_once(
        spatialdata_io.visium,
        path,
        zarr_path,
        dataset_id=dataset_id,
        **reader_kwargs,
    )


def visium_hd(
    path,
    *,
    dataset_id=None,
    bin_size=8,
    zarr_path=None,
    **reader_kwargs,
) -> spatialdata.SpatialData:
    reader_kwargs.setdefault("load_segmentations_only", False)
    return _read_once(
        spatialdata_io.visium_hd,
        path,
        zarr_path,
        dataset_id=dataset_id,
        bin_size=bin_size,
        **reader_kwargs,
    )


def xenium(
    path,
    *,
    aligned_images=True,
    zarr_path=None,
    **reader_kwargs,
) -> spatialdata.SpatialData:
    """A Xenium or Atera output folder as SpatialData.

    Read with `spatialdata_io.xenium`, except for the aligned H&E and IF images:
    spatialdata-io guesses their axes from the array shape and reads the whole slide
    as one chunk, which fails on the Atera H&E (scverse/spatialdata-io#168), so they
    are read with `_aligned_image` instead, before any zarr copy is written.

    The table's `obsm["spatial"]` holds cell centroids in microns; the "global"
    coordinate system is in morphology pixels of `xenium_pixel_size(path)` microns.

    Args:
        path: The output folder, with `experiment.xenium`.
        aligned_images: Load the aligned images in the folder (`*he_image.ome.tif`,
            `*if_image.ome.tif`) as `sdata.images["he_image"]` and `["if_image"]`.
        zarr_path: Write the result here once and read from it afterwards.
        reader_kwargs: Passed to `spatialdata_io.xenium`, e.g. `transcripts=False`.
    """
    return _read_once(
        _xenium,
        path,
        zarr_path,
        aligned_images=aligned_images,
        **reader_kwargs,
    )


def xenium_pixel_size(path) -> float:
    """Microns per morphology-image pixel: the unit of the "global" coordinate system
    in a Xenium or Atera SpatialData (`pixel_size` in `experiment.xenium`)."""
    specs = json.loads((pathlib.Path(path) / "experiment.xenium").read_text())
    return float(specs["pixel_size"])


def attach_spatial(
    adata: anndata.AnnData,
    spatial_dir: pathlib.Path | str,
    library_id: str,
    use_quality: str = "hires",
    image_dir: pathlib.Path | str | None = None,
) -> anndata.AnnData:
    """Add the tissue images and scale factors stLearn/scanpy/etc needs from a Visium
    table.

    Tables read with spatialdata-io keep the images in the SpatialData object,
    so this reads them, with `scalefactors_json.json`, from a Space Ranger
    `spatial` folder into `adata.uns["spatial"][library_id]`, then sets
    `obs["imagecol"]` and `obs["imagerow"]` as `stlearn.convert_scanpy` does.

    For Xenium and Atera, whose image is in the SpatialData object, use
    `attach_spatial_sdata`.

    Args:
        adata: Table with spot or bin coordinates in `obsm["spatial"]` at full
            resolution, e.g. `sdata.tables["table"]`.
        spatial_dir: Space Ranger `spatial` folder, with `scalefactors_json.json`
            and the tissue PNGs.
        library_id: Key under `adata.uns["spatial"]`, usually the sample ID.
        use_quality: Image the coordinates are scaled to, "hires" or "lowres".
        image_dir: Folder with `tissue_hires_image.png` and `tissue_lowres_image.png`,
            when they are not in `spatial_dir`.

    Returns:
        `adata`, updated in place.
    """
    spatial_dir = pathlib.Path(spatial_dir)
    image_dir = pathlib.Path(image_dir or spatial_dir)
    scalefactors = visium_scalefactors(
        json.loads((spatial_dir / "scalefactors_json.json").read_text())
    )
    images = {
        res: read_image(image_dir / f"tissue_{res}_image.png")
        for res in ("hires", "lowres")
        if (image_dir / f"tissue_{res}_image.png").exists()
    }
    # Keep the metadata spatialdata-io stored (chemistry, software version).
    previous: dict[str, Any] = next(iter(adata.uns.get("spatial", {}).values()), {})
    adata.uns["spatial"] = {
        library_id: {
            "images": images,
            "scalefactors": scalefactors,
            "metadata": previous.get("metadata", {}),
        }
    }
    return convert_scanpy(adata, use_quality=use_quality)


def visium_scalefactors(scalefactors: dict) -> dict:
    """Space Ranger's scale factors with `microns_per_pixel`: kept if present
    (Visium HD), otherwise 55 µm / `spot_diameter_fullres` (standard Visium)."""
    diameter = scalefactors.get("spot_diameter_fullres")
    if "microns_per_pixel" in scalefactors or not diameter:
        return scalefactors
    return dict(scalefactors, microns_per_pixel=VISIUM_SPOT_DIAMETER_UM / diameter)


def attach_spatial_sdata(
    adata: anndata.AnnData,
    sdata: spatialdata.SpatialData,
    library_id: str,
    microns_per_unit: float,
    *,
    image_key: str | None = None,
    coordinate_system: str = "global",
    spatial_units: Literal["micron", "coordinate_system"] = "micron",
    use_quality: str = "hires",
    max_px: int | None = 4000,
    spot_diameter_um: float = 10.0,
) -> anndata.AnnData:
    """Add the tissue image and scale factors stLearn needs to a table in `sdata`.

    Sets the same fields as `attach_spatial` and `read_10x`, with coordinates in
    pixels of a full-resolution image, as for Visium:

    - `obsm["spatial"]`: the cell centroids in full-resolution pixels. With
      `image_key`, those are the pixels of that image, e.g. the aligned H&E;
      without one, the units of `coordinate_system` (morphology-image pixels for
      Xenium and Atera "global").
    - `uns["spatial"][library_id]`: `images[use_quality]` (that image downsampled
      so its longest side is at most `max_px`, uint8 RGB),
      `use_quality`, and `scalefactors` with `tissue_{use_quality}_scalef` (image
      pixels per full-resolution pixel), `spot_diameter_fullres` (pixels) and
      `microns_per_pixel` (microns per full-resolution pixel).

    Distances in microns are full-resolution pixels times `microns_per_pixel`.

    Calling it again on the same table, e.g. for another image or `max_px`, starts
    from the cells' positions in `coordinate_system`, so they aren't converted
    twice.

    Args:
        adata: Table with cell centroids in `obsm["spatial"]`, e.g. a filtered copy
            of `sdata.tables["table"]`.
        sdata: The SpatialData the table came from (cropped or not); only its image
            and transformations are used. A cropped image's pixels start at the crop.
        library_id: Key under `adata.uns["spatial"]`, usually the sample ID.
        microns_per_unit: Microns per unit of `coordinate_system`. For Xenium and
            Atera "global" that is the morphology pixel size: `xenium_pixel_size`.
        image_key: Image in `sdata.images` to show behind the cells, e.g. "he_image".
            None gives a blank white image.
        coordinate_system: Coordinate system the image and cells are aligned in.
        spatial_units: Units of the input `obsm["spatial"]`: "micron" (Xenium,
            Atera) or "coordinate_system" (already in `coordinate_system` units).
        use_quality: Name of the image resolution, "hires" or "lowres".
        max_px: Longest side of the image, in pixels. The image is uint8 RGB:
            4000 x 4000 is about 48 MB, and grids made from the table copy it.
        spot_diameter_um: Typical cell diameter, used where stLearn needs a spot size
            (`st.tl.cci.run` with `distance=None` uses twice this).

    Returns:
        `adata`, updated in place, `obsm["spatial"]` included.
    """
    if spatial_units not in ("micron", "coordinate_system"):
        raise ValueError(
            "spatial_units must be 'micron' or 'coordinate_system', "
            f"not {spatial_units!r}"
        )
    xy = numpy.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
    previous: dict[str, Any] = next(iter(adata.uns.get("spatial", {}).values()), {})
    previous_metadata = previous.get("metadata", {})

    xy_cs = _attached_cells(previous_metadata, xy, coordinate_system)
    if xy_cs is None:
        xy_cs = xy / microns_per_unit if spatial_units == "micron" else xy

    if image_key is None:
        if (xy_cs < 0).any():
            raise ValueError(
                "obsm['spatial'] has negative coordinates; the blank image starts at "
                "the origin, so those cells would fall outside it."
            )
        # Full resolution is the coordinate system, from its origin to the furthest
        # cell.
        fullres_xy = xy_cs
        to_cs = Identity().to_affine_matrix(
            input_axes=("x", "y"), output_axes=("x", "y")
        )
        microns_per_pixel = float(microns_per_unit)
        full_size = numpy.floor(xy_cs.max(axis=0)) + 1  # width, height
    else:
        element = sdata.images[image_key]
        to_cs = _image_to_cs(element, image_key, coordinate_system)
        fullres_xy = _apply_affine(numpy.linalg.inv(to_cs), xy_cs)
        # Image pixels can be anisotropic or sheared; take the mean side length.
        microns_per_pixel = float(
            microns_per_unit * numpy.sqrt(abs(numpy.linalg.det(to_cs[:2, :2])))
        )
        level0 = _pyramid_levels(element)[0]
        full_size = numpy.array([level0.sizes["x"], level0.sizes["y"]], dtype=float)

    scalef = 1.0 if max_px is None else min(1.0, max_px / full_size.max())
    width, height = numpy.maximum(numpy.round(full_size * scalef), 1).astype(int)
    if image_key is None:
        image = numpy.full((height, width, 3), 255, dtype=numpy.uint8)
    else:
        image = _downsample(element, width, height)

    metadata = dict(
        previous_metadata,
        coordinate_system=coordinate_system,
        # So a later call can take obsm["spatial"] back to coordinate_system.
        fullres_to_coordinate_system=to_cs,
    )
    metadata.pop("image_key", None)
    if image_key is not None:
        metadata["image_key"] = image_key
    adata.obsm["spatial"] = fullres_xy
    adata.uns["spatial"] = {
        library_id: {
            "images": {use_quality: image},
            "use_quality": use_quality,
            "scalefactors": {
                f"tissue_{use_quality}_scalef": float(scalef),
                "spot_diameter_fullres": float(spot_diameter_um / microns_per_pixel),
                "microns_per_pixel": microns_per_pixel,
            },
            "metadata": metadata,
        }
    }
    return convert_scanpy(adata, use_quality=use_quality)


def save_table(sdata: spatialdata.SpatialData, name: str) -> None:
    """Write spatial data table `name`into the store, replacing the stored copy.

    SpatialData does not allow elements to be overwritten in place so they must
    be deleted and then written.
    """
    if sdata.path is None:
        raise ValueError(
            "sdata is not backed by a zarr store: read it with a `zarr_path`, "
            "or call sdata.write(path) first."
        )
    if name not in sdata.tables:
        raise KeyError(f"{name!r} is not a table of sdata: {sorted(sdata.tables)}")
    if f"tables/{name}" in sdata.elements_paths_on_disk():
        sdata.delete_element_from_disk(name)
    sdata.write_element(name)


def _read_once(reader, path, zarr_path, **reader_kwargs) -> spatialdata.SpatialData:
    """Read with `reader`; with `zarr_path`, write there once and read from it
    afterwards."""
    if zarr_path is not None and pathlib.Path(zarr_path).exists():
        return spatialdata.read_zarr(zarr_path)
    sdata = reader(path, **reader_kwargs)
    if zarr_path is not None:
        sdata.write(zarr_path)
    return sdata


def _xenium(path, *, aligned_images, **reader_kwargs) -> spatialdata.SpatialData:
    sdata = spatialdata_io.xenium(path, aligned_images=False, **reader_kwargs)
    if aligned_images:
        for name, image in _aligned_images(pathlib.Path(path)).items():
            sdata.images[name] = image
    return sdata


def _aligned_images(folder: pathlib.Path) -> dict:
    """The aligned images in a Xenium folder, read with `_aligned_image`.

    Found as `spatialdata_io.xenium(aligned_images=True)` finds them: files ending
    in "he_image.ome.tif" or "if_image.ome.tif", keyed "he_image" or "if_image",
    each with an optional "<file name without .ome.tif>alignment.csv" beside it.
    """
    images = {}
    for path in sorted(folder.glob("*.ome.tif")):
        for name in ("he_image", "if_image"):
            if path.name.endswith(f"{name}.ome.tif"):
                alignment = folder / path.name.replace(".ome.tif", "alignment.csv")
                images[name] = _aligned_image(
                    path,
                    alignment if alignment.exists() else None,
                    rgb=name == "he_image",
                )
    return images


def _aligned_image(
    image_path: pathlib.Path | str,
    alignment_path: pathlib.Path | str | None,
    rgb: bool = True,
    chunk: int = 4096,
):
    """An aligned H&E/IF OME-TIFF as a lazy, tiled image on the Xenium pixel grid.

    Reads tile by tile through tifffile's zarr interface and takes the axes from
    the OME metadata, so it works whatever the image's layout. Matches
    `spatialdata_io.xenium_aligned_image`: the alignment CSV is the 3x3 affine
    from image pixels to morphology-image pixels (no CSV: already aligned), and
    with `rgb` 3 or 4 channels are named r, g, b(, a).
    """
    with tifffile.TiffFile(image_path) as tif:
        axes = tif.series[0].axes
    store = tifffile.imread(image_path, aszarr=True, series=0, level=0)
    image = xr.DataArray(da.from_zarr(store), dims=list(axes))
    # Drop singleton axes (T, Z, a channel axis of 1, ...); keep Y, X and channels.
    singletons = [d for d in image.dims if d not in "YX" and image.sizes[d] == 1]
    image = image.squeeze(singletons)
    channels = [d for d in image.dims if d not in "YX"]
    if len(channels) > 1:
        raise ValueError(
            f"Can't read {image_path}: axes {axes!r}, shape {image.shape}",
        )
    image = image.rename({channels[0]: "c"}) if channels else image.expand_dims("c")
    image = image.rename({"Y": "y", "X": "x"}).transpose("c", "y", "x")

    n_channels = image.sizes["c"]
    if alignment_path is None:
        transformation = Identity()
    else:
        alignment = pd.read_csv(alignment_path, header=None).to_numpy()
        transformation = Affine(
            alignment, input_axes=("x", "y"), output_axes=("x", "y")
        )
    return Image2DModel.parse(
        image.data,
        dims=("c", "y", "x"),
        c_coords=list("rgba"[:n_channels]) if rgb and n_channels in (3, 4) else None,
        transformations={"global": transformation},
        chunks=(n_channels, chunk, chunk),
        scale_factors=[2, 2, 2, 2],
    )


def _image_to_cs(element, image_key, coordinate_system) -> numpy.ndarray:
    """3x3 affine from the image's level-0 pixels to `coordinate_system`."""
    systems = get_transformation(element, get_all=True)
    if coordinate_system not in systems:
        raise ValueError(
            f"Image {image_key!r} has no transformation to {coordinate_system!r}; "
            f"it has {sorted(systems)}."
        )
    return systems[coordinate_system].to_affine_matrix(
        input_axes=("x", "y"), output_axes=("x", "y")
    )


def _attached_cells(
    metadata: dict, xy: numpy.ndarray, coordinate_system: str
) -> numpy.ndarray | None:
    """Check for `obsm["spatial"]` (xy) already exists"""
    to_cs = metadata.get("fullres_to_coordinate_system")
    if to_cs is None:
        if "coordinate_system" in metadata:
            raise ValueError(
                "adata was attached before, so obsm['spatial'] is in pixels "
                "without a transform back to its coordinate system."
            )
        return None
    if metadata["coordinate_system"] != coordinate_system:
        raise ValueError(
            f"adata was attached in {metadata['coordinate_system']!r}, not "
            f"{coordinate_system!r}."
        )
    return _apply_affine(numpy.asarray(to_cs, dtype=float), xy)


def _apply_affine(matrix: numpy.ndarray, xy: numpy.ndarray) -> numpy.ndarray:
    """(n, 2) points through a 3x3 affine."""
    return xy @ matrix[:2, :2].T + matrix[:2, 2]


def _pyramid_levels(element) -> list[xr.DataArray]:
    """An image's resolution levels, largest first; one for a single-scale image."""
    if isinstance(element, xr.DataTree):
        return spatialdata.get_pyramid_levels(element)
    return [element]


def _downsample(element, width: int, height: int) -> numpy.ndarray:
    """The whole image resized to `height` x `width`, as uint8 RGB.

    Reads the smallest pyramid level at least that size and averages it down by a
    whole factor before the final resize, so memory stays near the output size."""
    levels = _pyramid_levels(element)
    level = next(
        (
            lv
            for lv in reversed(levels)
            if lv.sizes["x"] >= width and lv.sizes["y"] >= height
        ),
        levels[0],
    )
    factor = int(min(level.sizes["x"] / width, level.sizes["y"] / height))
    if factor > 1:
        level = (
            level.coarsen(x=factor, y=factor, boundary="trim")
            .mean()
            .round()
            .astype(level.dtype)
        )
    # Computes the dask graph: reads the chosen level tile by tile.
    rgb = _as_rgb(numpy.asarray(level.transpose("c", "y", "x").data))
    if rgb.shape[:2] != (height, width):
        # Resized as float32 in 0-255 (skimage would take uint8 to float64, twice
        # the memory), then rounded back to uint8.
        resized = resize(
            rgb.astype(numpy.float32),
            (height, width),
            order=1,
            anti_aliasing=True,
            preserve_range=True,
        )
        rgb = numpy.clip(numpy.rint(resized), 0, 255).astype(numpy.uint8)
    return rgb


def _as_rgb(pixels: numpy.ndarray) -> numpy.ndarray:
    """(c, y, x) image of any dtype as (y, x, 3) uint8."""
    if pixels.shape[0] >= 3:
        pixels = pixels[:3]  # drop alpha
    else:
        pixels = numpy.repeat(pixels[:1], 3, axis=0)  # one channel (IF, DAPI) as grey
    if pixels.dtype != numpy.uint8:
        # Other dtypes (uint16 IF, float) have no fixed range: stretch to the 99.9th
        # percentile of the non-zero pixels.
        pixels = pixels.astype(numpy.float32)
        nonzero = pixels[pixels > 0]
        top = numpy.percentile(nonzero, 99.9) if nonzero.size else 1.0
        pixels = as_uint8_image(pixels / top)
    return numpy.ascontiguousarray(pixels.transpose(1, 2, 0))
