import dataclasses
from typing import Literal

import anndata
import spatialdata
import spatialdata.models
import spatialdata.transformations
import xarray


def get_adata(data, table_key="table"):
    """Extract AnnData from either SpatialData or AnnData input."""
    if isinstance(data, spatialdata.SpatialData):
        return data.tables[table_key]
    elif isinstance(data, anndata.AnnData):
        return data
    else:
        raise TypeError(f"Expected SpatialData or AnnData, got {type(data)}")


def is_spatial_data(data):
    return isinstance(data, spatialdata.SpatialData)


@dataclasses.dataclass(frozen=True)
class SpatialElements:
    """Names of the SpatialData elements that go with one table."""

    table: str
    region: str  # the shapes or labels element the table annotates
    image: str
    coordinate_system: str


def is_fluorescence(image: xarray.DataArray | xarray.DataTree) -> bool:
    """Whether an image's channels are named stains (e.g. DAPI) rather than RGB."""
    rgb = {"r", "g", "b", "a"}
    return any(
        isinstance(c, str) and c not in rgb
        for c in spatialdata.models.get_channel_names(image)
    )


def _n_pixels(image: xarray.DataArray | xarray.DataTree) -> int:
    """Pixels in the full-resolution level of an image."""
    shape = (
        spatialdata.get_pyramid_levels(image, attr="shape", n=0)
        if isinstance(image, xarray.DataTree)
        else image.shape
    )
    return shape[-2] * shape[-1]


def _element(
    sdata: spatialdata.SpatialData, name: str
) -> spatialdata.models.SpatialElement:
    """The shapes, labels, image or points element called `name`."""
    element = sdata[name]
    if isinstance(element, anndata.AnnData):
        raise ValueError(f"{name!r} is a table, not a spatial element")
    return element


def element_transformations(
    element: spatialdata.models.SpatialElement,
) -> dict[str, spatialdata.transformations.BaseTransformation]:
    """An element's transformations, keyed by coordinate system."""
    transformations = spatialdata.transformations.get_transformation(
        element, get_all=True
    )
    if not isinstance(transformations, dict):  # never: get_all=True returns a dict
        raise TypeError(f"Expected a dict of transformations, got {transformations}")
    return transformations


def _is_unscaled(
    transformation: spatialdata.transformations.BaseTransformation,
) -> bool:
    """Does the transformation keep the element at its own pixel resolution?

    Identity and translations do; so does a sequence of them, which is what
    cropping a SpatialData object leaves on its elements.
    """
    if isinstance(transformation, spatialdata.transformations.Sequence):
        return all(map(_is_unscaled, transformation.transformations))
    return isinstance(
        transformation,
        spatialdata.transformations.Identity | spatialdata.transformations.Translation,
    )


def spatial_elements(
    sdata: spatialdata.SpatialData,
    table_key: str = "table",
    resolution: Literal["hires", "lowres"] = "lowres",
    image: str | None = None,
    coordinate_system: str | None = None,
) -> SpatialElements:
    """Find the element a table annotates, a tissue image, and
    a coordinate system for both.

    The image defaults to one sharing a coordinate system with the table's
    element, preferring brightfield (RGB, e.g. H&E) over fluorescence, then the
    smallest image for "lowres" or the largest for "hires". The coordinate
    system defaults to the one the image and the element share, preferring the
    image's native resolution.
    """
    region = sdata.tables[table_key].uns["spatialdata_attrs"]["region"]
    regions = [region] if isinstance(region, str) else list(region)
    if len(regions) != 1:
        raise ValueError(f"Table {table_key!r} annotates several elements: {regions}")
    region = regions[0]
    region_cs = element_transformations(_element(sdata, region)).keys()

    if image is None:
        images = list(
            filter(
                lambda name: (
                    region_cs & element_transformations(sdata.images[name]).keys()
                ),
                sdata.images,
            )
        )
        if not images:
            raise ValueError(
                f"No image shares a coordinate system with {region!r}; pass `image`"
            )
        size_order = 1 if resolution == "lowres" else -1
        image = min(
            images,
            key=lambda name: (
                is_fluorescence(sdata.images[name]),
                size_order * _n_pixels(sdata.images[name]),
            ),
        )

    if coordinate_system is None:
        image_transformations = element_transformations(sdata.images[image])
        shared = sorted(region_cs & image_transformations.keys())
        native = list(
            filter(lambda cs: _is_unscaled(image_transformations[cs]), shared)
        )
        options = native or shared
        if len(options) != 1:
            raise ValueError(
                f"Can't choose a coordinate system for {image!r} and {region!r}; "
                f"pass `coordinate_system`, one of {shared}"
            )
        coordinate_system = options[0]

    return SpatialElements(table_key, region, image, coordinate_system)
