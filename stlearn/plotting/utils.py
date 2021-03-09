import numpy as np
import pandas as pd

import io
from PIL import Image

import matplotlib
from anndata import AnnData

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


def centroidpython(x,y):
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

def check_sublist(full,sub):
    index_bool = []
    for barcode in full:
        if barcode in sub:
            index_bool.append(True)
        else:
            index_bool.append(False)
    return index_bool