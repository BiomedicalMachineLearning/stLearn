from collections.abc import Sequence
from typing import Literal

import matplotlib.axes
import spatialdata
import spatialdata.models
import spatialdata.transformations
import spatialdata_plot  # noqa: F401  (adds the .pl accessor to SpatialData)

from stlearn._compat import is_fluorescence, spatial_elements


def sdata_plot(
    sdata: spatialdata.SpatialData,
    color: str,
    *,
    table_key: str = "table",
    resolution: Literal["hires", "lowres"] = "lowres",
    image: str | None = None,
    channel: int | str | None = None,
    coordinate_system: str | None = None,
    groups: str | Sequence[str] | None = None,
    palette: str | Sequence[str] | None = None,
    fill_alpha: float = 0.7,
    method: Literal["matplotlib", "datashader"] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Plot a table column or gene over the tissue image of a SpatialData object.

    Works on data read with spatialdata-io: Visium spots, Visium HD bins and
    cell segmentations, and Xenium cells.

    Args:
        sdata: SpatialData object, e.g. from `st.datasets.visium_hd_sge`.
        color: Column in the table's `obs`, or a gene name.
        table_key: Table to plot: "table" for Visium and Xenium, "square_008um"
            or "cell_segmentations" for Visium HD.
        resolution: "lowres" plots on the smallest tissue image, "hires" on
            the largest. Brightfield (H&E) images are preferred either way.
        image: Image to plot on, instead of choosing one.
        channel: Image channel to show. Defaults to the first channel (DAPI)
            for fluorescence images such as Xenium's `morphology_focus`.
        coordinate_system: Coordinate system to plot in, if it can't be
            worked out from the image and the table.
        groups: Only colour these categories of `color`.
        palette: Colours for the categories in `groups`.
        fill_alpha: Opacity of the spots, bins or cells.
        method: How to draw shapes; "datashader" is much faster for large
            tables. Chosen automatically when not given.
        title: Plot title. Defaults to `color`.
        figsize: Figure size.
        ax: Axes to draw on.

    Returns:
        The axes plotted on.
    """
    elements = spatial_elements(sdata, table_key, resolution, image, coordinate_system)

    if channel is None and is_fluorescence(sdata.images[elements.image]):
        channel = spatialdata.models.get_channel_names(sdata.images[elements.image])[0]

    plot = sdata.pl.render_images(
        elements.image,
        channel=channel,
        cmap="gray" if channel is not None else None,
        colorbar=False,
    )
    style = {
        "color": color,
        "table_name": table_key,
        "groups": groups,
        "palette": palette,
        "fill_alpha": fill_alpha,
    }

    if elements.region in sdata.labels:
        return plot.pl.render_labels(elements.region, **style).pl.show(
            coordinate_systems=elements.coordinate_system,
            title=title or color,
            figsize=figsize,
            ax=ax,
            return_ax=True,
        )

    # spatialdata-plot 0.4.0 sizes the datashader canvas in a "global" coordinate
    # system, which spatialdata-io's Visium elements don't have. Add an identity
    # one for the duration of the plot.
    shapes = sdata.shapes[elements.region]
    add_global = "global" not in spatialdata.transformations.get_transformation(
        shapes, get_all=True
    )
    if add_global:
        spatialdata.transformations.set_transformation(
            shapes, spatialdata.transformations.Identity(), "global"
        )
    try:
        return plot.pl.render_shapes(elements.region, method=method, **style).pl.show(
            coordinate_systems=elements.coordinate_system,
            title=title or color,
            figsize=figsize,
            ax=ax,
            return_ax=True,
        )
    finally:
        if add_global:
            spatialdata.transformations.remove_transformation(shapes, "global")
