import numpy
import spatialdata
from spatialdata.transformations import get_transformation


def crop_visium_hd(sdata, sample_id, table_name, box=None):
    """`sdata` cut to `box`, a (lo, hi) pair of (x, y) corners in full-resolution
    pixels; by default a window in the middle of the sample."""
    if box is None:
        xy = sdata.tables[table_name].obsm["spatial"]
        box = window((xy.min(0), xy.max(0)))
    return _crop(sdata, *box, sample_id)


def crop_xenium(sdata, table_name, coordinate_system="global", box=None):
    """`sdata` cut to `box`, a (lo, hi) pair of (x, y) corners in morphology-image
    pixels; by default a window in the middle of the cells."""
    if box is None:
        box = window(cells_extent(sdata, table_name, coordinate_system))
    return _crop(sdata, *box, coordinate_system)


def window(extent, centre=(0.5, 0.5), size=(0.25, 0.25)):
    """The (lo, hi) box of `size` centred on `centre`, both as (x, y) fractions of
    `extent` (itself a (lo, hi) pair), in the extent's units."""
    lo, hi = extent
    span = hi - lo
    centre = lo + span * numpy.asarray(centre)
    half = span * numpy.asarray(size) / 2
    return centre - half, centre + half


def cells_extent(sdata, table_name, coordinate_system):
    """Lower and upper (x, y) corners of the cells (or bins) `table_name` annotates."""
    region = sdata.tables[table_name].uns["spatialdata_attrs"]["region"]
    regions = [region] if isinstance(region, str) else list(region)
    ext = spatialdata.get_extent(
        sdata, coordinate_system=coordinate_system, elements=regions
    )
    return numpy.array([ext["x"][0], ext["y"][0]]), numpy.array(
        [ext["x"][1], ext["y"][1]]
    )


def crop_to_cells(sdata, table_name, coordinate_system="global", pad=500):
    """`sdata` cut to the box around its cells, dropping tissue the image shows but
    the assay didn't measure. `pad` is a margin in the coordinate system's units."""
    lo, hi = cells_extent(sdata, table_name, coordinate_system)
    return _crop(sdata, lo - pad, hi + pad, coordinate_system)


def _crop(sdata, lo, hi, coordinate_system):
    """`sdata` cut to the box from `lo` to `hi`, with images, labels and points left
    lazy.

    Points are cropped here because `bounding_box_query` loads every point into
    memory first (624 million transcripts on the Atera slide). If the table annotates
    labels rather than shapes, `bounding_box_query` reads the cropped labels' pixels
    to find which cells to keep."""

    without_points = spatialdata.SpatialData(
        images=dict(sdata.images),
        labels=dict(sdata.labels),
        shapes=dict(sdata.shapes),
        tables=dict(sdata.tables),
        attrs=sdata.attrs,
    )
    cropped = spatialdata.bounding_box_query(
        without_points,
        axes=("x", "y"),
        min_coordinate=lo,
        max_coordinate=hi,
        target_coordinate_system=coordinate_system,
    )
    for name, points in sdata.points.items():
        if coordinate_system in get_transformation(points, get_all=True):
            cropped.points[name] = _crop_points(points, lo, hi, coordinate_system)
    return cropped


def _crop_points(points, lo, hi, coordinate_system):
    """The points inside the box from `lo` to `hi`, as a lazy filter."""
    m = get_transformation(points, coordinate_system).to_affine_matrix(
        input_axes=("x", "y"), output_axes=("x", "y")
    )
    x = m[0, 0] * points["x"] + m[0, 1] * points["y"] + m[0, 2]
    y = m[1, 0] * points["x"] + m[1, 1] * points["y"] + m[1, 2]
    cropped = points[(x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])]
    cropped.attrs.update(points.attrs)  # transformations and spatialdata_attrs
    return cropped
