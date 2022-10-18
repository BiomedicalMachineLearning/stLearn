import numpy as np
import pandas as pd
import io
from PIL import Image
import matplotlib
from anndata import AnnData
import networkx as nx

from typing import Optional, Union, Mapping  # Special
from typing import Sequence, Iterable  # ABCs
from typing import Tuple  # Classes

from textwrap import dedent

from enum import Enum


class Empty(Enum):
    token = 0


_empty = Empty.token

from matplotlib import rcParams, ticker, gridspec, axes
from matplotlib.axes import Axes
from abc import ABC


class _AxesSubplot(Axes, axes.SubplotBase, ABC):
    """Intersection between Axes and SubplotBase: Has methods of both"""


def _check_spot_size(
    spatial_data: Optional[Mapping], spot_size: Optional[float]
) -> float:
    """
    Resolve spot_size value.
    This is a required argument for spatial plots.
    """
    if spatial_data is None and spot_size is None:
        raise ValueError(
            "When .uns['spatial'][library_id] does not exist, spot_size must be "
            "provided directly."
        )
    elif spot_size is None:
        return spatial_data["scalefactors"]["spot_diameter_fullres"]
    else:
        return spot_size


def _check_scale_factor(
    spatial_data: Optional[Mapping],
    img_key: Optional[str],
    scale_factor: Optional[float],
) -> float:
    """Resolve scale_factor, defaults to 1."""
    if scale_factor is not None:
        return scale_factor
    elif spatial_data is not None and img_key is not None:
        return spatial_data["scalefactors"][f"tissue_{img_key}_scalef"]
    else:
        return 1.0


def _check_spatial_data(
    uns: Mapping, library_id: Union[Empty, None, str]
) -> Tuple[Optional[str], Optional[Mapping]]:
    """
    Given a mapping, try and extract a library id/ mapping with spatial data.
    Assumes this is `.uns` from how we parse visium data.
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
    spatial_data: Optional[Mapping],
    img: Optional[np.ndarray],
    img_key: Union[None, str, Empty],
    bw: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Resolve image for spatial plots.
    """
    if img is None and spatial_data is not None and img_key is _empty:
        img_key = next(
            (k for k in ["hires", "lowres", "fulres"] if k in spatial_data["images"]),
        )  # Throws StopIteration Error if keys not present
    if img is None and spatial_data is not None and img_key is not None:
        img = spatial_data["images"][img_key]
    if bw:
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img, img_key


def _check_coords(
    obsm: Optional[Mapping], scale_factor: Optional[float]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

    image_coor = obsm["spatial"] * scale_factor
    imagecol = image_coor[:, 0]
    imagerow = image_coor[:, 1]

    return [imagecol, imagerow]


def _read_graph(adata: AnnData, graph_type: Optional[str]):

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
