from collections.abc import Mapping
from enum import Enum
from textwrap import dedent

import networkx as nx
import numpy as np
from anndata import AnnData
from matplotlib import axes
from matplotlib.axes import Axes


class Empty(Enum):
    token = 0


_empty = Empty.token


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
            "When .uns['spatial'][library_id] does not exist, spot_size must be "
            "provided directly."
        )

    return spatial_data["scalefactors"]["spot_diameter_fullres"]


def _check_scale_factor(
    spatial_data: Mapping | None,
    img_key: str | None,
    scale_factor: float | None,
) -> float:
    """Resolve scale_factor, defaults to 1."""
    if scale_factor is not None:
        return scale_factor
    elif spatial_data is not None and img_key is not None:
        return spatial_data["scalefactors"][f"tissue_{img_key}_scalef"]
    else:
        return 1.0


def _check_spatial_data(
    uns: Mapping, library_id: Empty | None | str
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
            library_id = list(spatial_mapping.keys())[0]
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
    img_key: None | str | Empty,
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


def _read_graph(adata: AnnData, graph_type: str | None):
    if graph_type == "PTS_graph":
        graph = nx.from_scipy_sparse_array(
            adata.uns[graph_type]["graph"], create_using=nx.DiGraph
        )
    else:
        graph = nx.from_scipy_sparse_array(adata.uns[graph_type]["graph"])
    node_dict = adata.uns[graph_type]["node_dict"]
    node_dict = {int(k): int(v) for k, v in node_dict.items()}

    relabel_graph = nx.relabel_nodes(graph, node_dict)

    return relabel_graph


def _docs_params(**kwds):
    """\
    Docstrings should start with "\" in the first line for proper formatting.
    """

    def dec(obj):
        obj.__orig_doc__ = obj.__doc__
        obj.__doc__ = dedent(obj.__doc__).format_map(kwds)
        return obj

    return dec
