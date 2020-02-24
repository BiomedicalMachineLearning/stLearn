from .model_zoo import encode
from typing import Optional, Union
from anndata import AnnData
import numpy as np
from .._compat import Literal
from PIL import Image
import pandas as pd
from pathlib import Path


_CNN_BASE = Literal['resnet50', 'vgg16', 'inceptionv3']


def extract_feature(
        adata: AnnData,
        cnn_base: _CNN_BASE = 'resnet50',
        copy: bool = False
) -> Optional[AnnData]:
    feature_df = pd.DataFrame()
    model = None
    if cnn_base == 'resnet50':
        from .model_zoo import ResNet
        model = ResNet()

    for spot, tile_path in adata.obs["tile_path"].items():
        tile = Image.open(tile_path)
        tile = np.asarray(tile, dtype="int32")
        tile = tile.astype(np.float32)
        tile = np.stack([tile])

        print("extract feature for spot: {}".format(str(spot)))
        features = encode(tile, model)
        feature_df[spot] = features

    adata.obsm["tile_feature"] = feature_df.transpose().to_numpy()

    return adata if copy else None

# TODO: add more CNN base
