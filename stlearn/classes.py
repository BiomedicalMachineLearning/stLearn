"""
Title: SpatialBasePlot for all spatial coordinates and image plot
Author: Duy Pham
Date: 20 Feb 2021
"""

from typing import Optional, Union, Mapping  # Special
from typing import Sequence, Iterable  # ABCs
from typing import Tuple  # Classes

import numpy as np
from anndata import AnnData

from .utils import (
    Empty,
    _empty,
    _check_spatial_data,
    _check_img,
    _check_spot_size,
    _check_scale_factor,
    _check_coords,
)


class Spatial(object):
    def __init__(
        self,
        adata: AnnData,
        basis: str = "spatial",
        img: Union[np.ndarray, None] = None,
        img_key: Union[str, None, Empty] = _empty,
        library_id: Union[str, None] = _empty,
        crop_coord: Optional[bool] = True,
        bw: Optional[bool] = False,
        scale_factor: Optional[float] = None,
        spot_size: Optional[float] = None,
        use_raw: Optional[bool] = False,
        **kwargs,
    ):

        self.adata = (adata,)
        self.library_id, self.spatial_data = _check_spatial_data(adata.uns, library_id)
        self.img, self.img_key = _check_img(self.spatial_data, img, img_key, bw=bw)
        self.spot_size = _check_spot_size(self.spatial_data, spot_size)
        self.scale_factor = _check_scale_factor(
            self.spatial_data, img_key=self.img_key, scale_factor=scale_factor
        )
        self.crop_coord = crop_coord
        self.use_raw = use_raw
        self.imagecol, self.imagerow = _check_coords(adata.obsm, self.scale_factor)
