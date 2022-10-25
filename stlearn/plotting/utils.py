import numpy as np
import pandas as pd

import io
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from anndata import AnnData
from scanpy.plotting import palettes
from stlearn.plotting import palettes_st

from typing import Optional, Union, Mapping  # Special
from typing import Sequence, Iterable  # ABCs
from typing import Tuple  # Classes

from enum import Enum

from matplotlib import rcParams, ticker, gridspec, axes
from matplotlib.axes import Axes
from abc import ABC


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    from io import BytesIO

    fig.savefig(
        buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0, transparent=True
    )
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = np.asarray(Image.open(BytesIO(img_arr)))
    buf.close()
    # img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    return img


def centroidpython(x, y):
    l = len(x)
    return sum(x) / l, sum(y) / l


def get_cluster(search, dictionary):
    for (
        cl,
        sub,
    ) in (
        dictionary.items()
    ):  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if search in sub:
            return cl


def get_node(node_list, split_node):
    result = np.array([])
    for node in node_list:
        result = np.append(result, np.array(split_node[node]).astype(int))
    return result.astype(int)


def check_sublist(full, sub):
    set_sub = set(sub)
    index_bool = [x in set_sub for x in full]

    return index_bool


def get_cmap(cmap):
    """Checks inputted cmap string."""
    if cmap == "vega_10_scanpy":
        cmap = palettes.vega_10_scanpy
    elif cmap == "vega_20_scanpy":
        cmap = palettes.vega_20_scanpy
    elif cmap == "default_102":
        cmap = palettes.default_102
    elif cmap == "default_28":
        cmap = palettes.default_28
    elif cmap == "jana_40":
        cmap = palettes_st.jana_40
    elif cmap == "default":
        cmap = palettes_st.default
    elif type(cmap) == str:  # If refers to matplotlib cmap
        cmap_n = plt.get_cmap(cmap).N
        return plt.get_cmap(cmap), cmap_n
    elif type(cmap) == matplotlib.colors.LinearSegmentedColormap:  # already cmap
        cmap_n = cmap.N
        return cmap, cmap_n

    cmap_n = len(cmap)
    cmaps = matplotlib.colors.LinearSegmentedColormap.from_list("", cmap)

    cmap_ = plt.cm.get_cmap(cmaps)

    return cmap_, cmap_n


def check_cmap(cmap):
    """Initialize cmap"""
    scanpy_cmap = ["vega_10_scanpy", "vega_20_scanpy", "default_102", "default_28"]
    stlearn_cmap = ["jana_40", "default"]
    cmap_available = plt.colormaps() + scanpy_cmap + stlearn_cmap
    error_msg = (
        "cmap must be a matplotlib.colors.LinearSegmentedColormap OR"
        "one of these: " + str(cmap_available)
    )
    if type(cmap) == str:
        assert cmap in cmap_available, error_msg
    elif type(cmap) != matplotlib.colors.LinearSegmentedColormap:
        raise Exception(error_msg)

    return cmap


def get_colors(adata, obs_key, cmap="default", label_set=None):
    """Retrieves colors if present in adata.uns, if not present then will set
    them as per scanpy & return in order requested.
    """
    # Checking if colors are already set #
    col_key = f"{obs_key}_colors"
    if col_key in adata.uns:
        labels_ordered = adata.obs[obs_key].cat.categories
        colors_ordered = adata.uns[col_key]
    else:  # Colors not already present
        check_cmap(cmap)
        cmap, cmap_n = get_cmap(cmap)

        if not hasattr(adata.obs[obs_key], "cat"):  # Ensure categorical
            adata.obs[obs_key] = adata.obs[obs_key].astype("category")
        labels_ordered = adata.obs[obs_key].cat.categories
        colors_ordered = [
            matplotlib.colors.rgb2hex(cmap(i / (len(labels_ordered) - 1)))
            for i in range(len(labels_ordered))
        ]
        adata.uns[col_key] = colors_ordered

    # Returning the colors of the desired labels in indicated order #
    if type(label_set) != type(None):
        colors_ordered = [
            colors_ordered[np.where(labels_ordered == label)[0][0]]
            for label in label_set
        ]

    return colors_ordered
