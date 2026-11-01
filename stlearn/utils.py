import warnings
from collections.abc import Iterable, Mapping
from enum import Enum
from textwrap import dedent

import networkx as nx
import numba
import numpy as np
import PIL.Image
from anndata import AnnData
from matplotlib import axes
from matplotlib.axes import Axes


class Empty(Enum):
    token = 0


_empty = Empty.token

# The root of the spatial trajectory in adata.uns["PTS_graph"]: it points at the
# subclusters of the first cluster. Subcluster labels are numbers, so none can
# be called "root".
_TRAJECTORY_ROOT = "root"


class _AxesSubplot(Axes, axes.SubplotBase):
    """Intersection between Axes and SubplotBase: Has methods of both"""


def _check_spot_size(spatial_data: Mapping | None, spot_size: float | None) -> float:
    """
    Resolve spot_size value.
    This is a required argument for spatial plots.
    """
    if spot_size is not None:
        return spot_size

    if spatial_data is None:
        raise ValueError(
            "adata has no .uns['spatial'][library_id]; tables read with spatialdata-io "
            "don't. Add it with stlearn.io.attach_spatial (Visium, Visium HD) or "
            "stlearn.io.attach_spatial_sdata (Xenium, Atera), or pass spot_size."
        )

    return spatial_data["scalefactors"]["spot_diameter_fullres"]


def _set_n_threads(n_cpus: int | None) -> None:
    """Sets numba's thread count, defaulting to the most it allows.

    numba.set_num_threads raises if asked for more than
    numba.config.NUMBA_NUM_THREADS (the core count, or the NUMBA_NUM_THREADS
    environment variable if set), so larger requests are clamped with a warning.
    """
    max_threads = numba.config.NUMBA_NUM_THREADS
    if n_cpus is None:
        n_cpus = max_threads
    elif n_cpus < 1:
        raise ValueError(f"n_cpus must be at least 1, got {n_cpus}")
    elif n_cpus > max_threads:
        warnings.warn(
            f"n_cpus={n_cpus} exceeds numba's limit of {max_threads} threads; "
            f"using {max_threads}.",
            stacklevel=3,
        )
        n_cpus = max_threads
    numba.set_num_threads(n_cpus)


def _check_scale_factor(
    spatial_data: Mapping | None,
    img_key: str | None,
    scale_factor: float | None,
) -> float:
    """Resolve scale_factor, defaults to 1."""
    if scale_factor is not None:
        return scale_factor
    elif spatial_data is not None and img_key is not None:
        return _quality_scalef(spatial_data, img_key)
    else:
        return 1.0


def _check_spatial_data(
    uns: Mapping, library_id: Empty | str | None
) -> tuple[str | Empty | None, Mapping | None]:
    """
    Given a mapping, try and extract a library id/ mapping with spatial data.
    Assumes this is `.uns` from how we parse visium data.

    Parameters
    ----------
    library_id : None | str | Empty
        If None - don't find an image. Empty - find best image, or specify with str.
    """
    spatial_mapping = uns.get("spatial", {})
    if library_id is _empty:
        if len(spatial_mapping) > 1:
            raise ValueError(
                "Found multiple possible libraries in `.uns['spatial']. Please specify."
                f" Options are:\n\t{list(spatial_mapping.keys())}"
            )
        elif len(spatial_mapping) == 1:
            library_id = next(iter(spatial_mapping.keys()))
        else:
            library_id = None
    if library_id is not None:
        spatial_data = spatial_mapping[library_id]
    else:
        spatial_data = None
    return library_id, spatial_data


def _check_img(
    spatial_data: Mapping | None,
    img: np.ndarray | None,
    img_key: str | Empty | None,
    bw: bool = False,
) -> tuple[np.ndarray | None, str | None]:
    """
    Resolve image for spatial plots.

    Parameters
    ----------
    img : np.ndarray | None
        If given an image will not look for another image and not check to see if it
        was in spatial_data.
    img_key : None | str | Empty
        If None - don't find an image. Empty - find best image, or specify with str.

    Returns
    -------
    tuple[np.ndarray | None, str | None]
        The image found or nothing, str of the key of image found or None if none found.


    """

    # Return [None, None] if there's no anndata mapping or img
    if spatial_data is None and img is None:
        return None, None
    else:
        # Find image and key
        new_img_key: str | None = None
        new_img: np.ndarray | None = None

        # Return the img if not None and convert the key to Empty -> None if Empty
        # otherwise keep.
        if img is not None:
            new_img = img
            new_img_key = img_key if img_key is not _empty else None
        # Find key if empty or use key.
        elif spatial_data is not None:
            if img_key is _empty:
                # Looks for image - or None if not found.
                new_img_key = next(
                    (
                        k
                        for k in ["hires", "lowres", "fulres"]
                        if k in spatial_data["images"]
                    ),
                    None,
                )
            else:
                new_img_key = img_key

            if new_img_key is not None:
                new_img = spatial_data["images"][new_img_key]

        if new_img is not None and bw:
            new_img = np.dot(new_img[..., :3], [0.2989, 0.5870, 0.1140])

        return new_img, new_img_key


def as_uint8_image(image: np.ndarray) -> np.ndarray:
    """An image as uint8, the dtype stLearn stores the images in `uns["spatial"]` in.

    uint8 images are returned as they are. Floats are taken to be in [0, 1], as
    matplotlib reads PNGs, and scaled to 0-255; other unsigned integers are scaled
    from their full range (0-65535 for uint16). Signed integers, as PIL opens some
    16-bit images (mode "I", int32), are clipped at 0 and scaled as uint8 would be
    if none is above 255, else as uint16 (values above 65535 clipped).
    """
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if image.dtype.kind == "f":
        scaled = np.clip(image, 0, 1, dtype=np.float32)
        scaled *= 255
    elif image.dtype.kind == "b":
        scaled = image.astype(np.float32) * 255
    elif image.dtype.kind == "u":
        scaled = image.astype(np.float32) * (255 / np.iinfo(image.dtype).max)
    elif image.dtype.kind == "i":
        top = 65535 if image.size and image.max() > 255 else 255
        scaled = np.clip(image, 0, top).astype(np.float32) * (255 / top)
    else:
        raise TypeError(f"Can't convert an image of {image.dtype} to uint8.")
    return np.rint(scaled, out=scaled).astype(np.uint8)


def read_image(path) -> np.ndarray:
    """An image file as uint8, the dtype stLearn stores images in `uns["spatial"]` in.

    16- and 32-bit greyscale are scaled to 0-255 by as_uint8_image;
    matplotlib's `imread` would clip 32-bit integers to 0-255. Palette and
    grey-with-alpha images become RGBA, other colour modes RGB.
    """
    with PIL.Image.open(path) as image:
        if image.mode in ("P", "PA", "LA"):
            image = image.convert("RGBA")
        elif image.mode in ("CMYK", "YCbCr", "LAB", "HSV"):
            image = image.convert("RGB")
        return as_uint8_image(np.array(image))


def microns_to_pixels(
    adata: AnnData,
    microns: float | np.ndarray,
    microns_per_pixel: float | None = None,
) -> float | np.ndarray:
    """Microns as the image pixels stLearn measures distances in.

    stLearn's distances and radii are in pixels of the image at `use_quality`,
    the units of `obs["imagecol"]` and `obs["imagerow"]`: `distance` in
    `st.tl.cci.run`, and `radius` in `st.spatial.morphology.adjust`,
    `st.spatial.trajectory.pseudotime` and `st.spatial.smooth.disk`. So
    `distance=st.utils.microns_to_pixels(grid, 270)` is 270 µm on any sample.

    Parameters
    ----------
    adata: AnnData
        The data object, with `uns["spatial"]`.
    microns: float or np.ndarray
        Lengths in microns.
    microns_per_pixel: float
        Microns per full-resolution pixel. If None, uses
        `uns["spatial"][library_id]["scalefactors"]["microns_per_pixel"]`, which
        the `st.io` readers set and Space Ranger writes for Visium HD.

    Returns
    -------
    The lengths in image pixels.

    Raises
    ------
    ValueError
        If there is no `microns_per_pixel`.
    """
    return microns / _microns_per_image_pixel(adata, microns_per_pixel)


def pixels_to_microns(
    adata: AnnData,
    pixels: float | np.ndarray,
    microns_per_pixel: float | None = None,
) -> float | np.ndarray:
    """Image pixels, as stLearn measures distances in, in microns; the inverse of
    `microns_to_pixels`, which describes the parameters."""
    return pixels * _microns_per_image_pixel(adata, microns_per_pixel)


def get_shape(
    adata: AnnData, bin_um: float, microns_per_pixel: float | None = None
) -> tuple[int, int]:
    """The grid of squares about `bin_um` microns across that st.tl.cci.create_grid
    grids adata into, as (n_col, n_row): each side of the tissue's extent in
    microns rounded to a whole number of squares, at least one.

    Parameters
    ----------
    adata: AnnData
        The data object, with obs["imagecol"], obs["imagerow"] and uns["spatial"].
    bin_um: float
        Side of a grid square in microns.
    microns_per_pixel: float
        Microns per full-resolution pixel; see `microns_to_pixels`.

    Returns
    -------
    (n_col, n_row): the columns and rows, as st.tl.cci.grid takes them.
    """
    um_per_image_px = _microns_per_image_pixel(adata, microns_per_pixel)
    xs, ys = _image_coords(adata)
    extent_um = np.array([np.ptp(xs), np.ptp(ys)]) * um_per_image_px
    n_col, n_row = np.maximum(np.round(extent_um / bin_um), 1).astype(int)
    return int(n_col), int(n_row)


def _image_coords(adata: AnnData) -> tuple[np.ndarray, np.ndarray]:
    """The cells' coordinates as grid() bins them: whole image pixels."""
    return (
        adata.obs["imagecol"].values.astype(int),
        adata.obs["imagerow"].values.astype(int),
    )


def _quality_scalef(spatial: Mapping, quality: str) -> float:
    """Image pixels per full-resolution pixel for one library's `quality` image,
    "hires", "lowres" or "fulres"; `spatial` is adata.uns["spatial"][library_id]."""
    if quality == "fulres":
        return 1.0
    return spatial["scalefactors"][f"tissue_{quality}_scalef"]


def _image_scalef(adata: AnnData) -> float:
    """Image pixels per full-resolution pixel, for the image at `use_quality`."""
    spatial = next(iter(adata.uns["spatial"].values()))
    return _quality_scalef(spatial, spatial["use_quality"])


def _microns_per_image_pixel(
    adata: AnnData, microns_per_pixel: float | None = None
) -> float:
    """Microns per pixel of the image at `use_quality`, the units of
    `obs["imagecol"]` and `obs["imagerow"]`."""
    if microns_per_pixel is None:
        library_id, spatial = next(iter(adata.uns["spatial"].items()))
        microns_per_pixel = spatial["scalefactors"].get("microns_per_pixel")
        if microns_per_pixel is None:
            raise ValueError(
                f"microns_per_pixel missing in {library_id!r} - pixels can't be "
                "converted to microns. Pass microns_per_pixel "
                "(microns per full-resolution pixel), or set it "
                f"in adata.uns['spatial'][{library_id!r}]['scalefactors']"
                "['microns_per_pixel']."
            )
    return microns_per_pixel / _image_scalef(adata)


def _check_coords(
    obsm: Mapping | None, scale_factor: float | None
) -> tuple[np.ndarray, np.ndarray]:
    if obsm is None:
        raise ValueError("obsm cannot be None")
    if scale_factor is None:
        raise ValueError("scale_factor cannot be None")
    if "spatial" not in obsm:
        raise ValueError("'spatial' key not found in obsm")

    image_coor = obsm["spatial"] * scale_factor
    imagecol = image_coor[:, 0]
    imagerow = image_coor[:, 1]

    return (imagecol, imagerow)


# Old files storing "node_dict" instead of "nodes" aren't supported;
# convert them first (see below).
def _read_graph(adata: AnnData, graph_type: str):
    entry = adata.uns[graph_type]
    if "nodes" not in entry:
        raise ValueError(
            "No graph location found ('nodes' in adata.uns). If using "
            "an old version of stLearn upgrade the node_dict if you have one."
        )
    create_using = nx.DiGraph if graph_type == "PTS_graph" else nx.Graph
    graph = nx.from_scipy_sparse_array(entry["graph"], create_using=create_using)

    nodes = [
        tuple(n) if np.ndim(n) else n
        for n in np.asarray(entry["nodes"], dtype=object).tolist()
    ]
    return nx.relabel_nodes(graph, dict(enumerate(nodes)))


def _read_trajectory(adata: AnnData) -> nx.DiGraph:
    """The spatial trajectory between subclusters, without its root."""
    graph = _read_graph(adata, "PTS_graph")
    graph.remove_node(_TRAJECTORY_ROOT)
    return graph


def _cluster_labels(
    adata: AnnData, use_label: str, clusters: str | Iterable[str]
) -> list[str]:
    """Return cluster labels, checked against adata.obs[use_label].

    Takes one label or several. Labels must be strings, as in the categories of
    adata.obs[use_label]: "6", not 6. A cluster is never chosen by its position.
    """
    categories = list(adata.obs[use_label].cat.categories)
    if not all(isinstance(category, str) for category in categories):
        raise TypeError(
            f"adata.obs[{use_label!r}] must have string categories. Convert it with "
            f"adata.obs[{use_label!r}] = adata.obs[{use_label!r}]"
            ".astype(str).astype('category')"
        )
    if isinstance(clusters, str) or not isinstance(clusters, Iterable):
        clusters = [clusters]
    not_strings = [cluster for cluster in clusters if not isinstance(cluster, str)]
    if not_strings:
        raise TypeError(
            "Cluster labels must be strings: use "
            f"{str(not_strings[0])!r} rather than {not_strings[0]!r}."
        )
    labels = [str(cluster) for cluster in clusters]
    missing = [label for label in labels if label not in categories]
    if missing:
        raise ValueError(
            f"Cluster(s) {missing} not found in adata.obs[{use_label!r}]. "
            f"Choose from: {categories}"
        )
    return labels


def _cluster_positions(
    adata: AnnData, use_label: str, clusters: str | Iterable[str]
) -> list[int]:
    """Positions of cluster labels in the categories of adata.obs[use_label].

    The positions are the row order of PAGA's connectivities and of
    adata.uns[f"{use_label}_colors"], and the keys of adata.uns["split_node"].
    """
    categories = adata.obs[use_label].cat.categories
    return [
        categories.get_loc(label)
        for label in _cluster_labels(adata, use_label, clusters)
    ]


def _check_cluster(adata: AnnData, use_label: str, cluster: str) -> str:
    """Return `cluster` as a label in adata.obs[use_label], or raise."""
    return _cluster_labels(adata, use_label, cluster)[0]


def _docs_params(**kwds):
    """\
    Docstrings should start with "\" in the first line for proper formatting.
    """

    def dec(obj):
        obj.__orig_doc__ = obj.__doc__
        obj.__doc__ = dedent(obj.__doc__).format_map(kwds)
        return obj

    return dec
