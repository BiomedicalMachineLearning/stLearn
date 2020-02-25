from .model_zoo import encode, Model
from typing import Optional, Union
from anndata import AnnData
import numpy as np
from .._compat import Literal
from PIL import Image
import pandas as pd
from pathlib import Path

# Test progress bar
from tqdm import tqdm

_CNN_BASE = Literal['resnet50', 'vgg16', 'inception_v3', 'xception']


def extract_feature(
        adata: AnnData,
        cnn_base: _CNN_BASE = 'resnet50',
        n_components: int = 50,
        verbose: bool = False,
        copy: bool = False
) -> Optional[AnnData]:
    feature_df = pd.DataFrame()
    model = Model(cnn_base)

    if "tile_path" not in adata.obs:
        raise ValueError("Please run the function stlearn.pp.tiling")

    with tqdm(total=len(adata), desc="Extract feature", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for spot, tile_path in adata.obs["tile_path"].items():
            tile = Image.open(tile_path)
            tile = np.asarray(tile, dtype="int32")
            tile = tile.astype(np.float32)
            tile = np.stack([tile])
            if verbose:
                print("extract feature for spot: {}".format(str(spot)))
            features = encode(tile, model)
            feature_df[spot] = features
            pbar.update(1)

    adata.obsm["X_tile_feature"] = feature_df.transpose().to_numpy()

    from sklearn.decomposition import PCA
    pca = PCA(n_components = n_components)
    pca.fit(feature_df.transpose().to_numpy())

    adata.obsm["X_morphology"] = pca.transform(feature_df.transpose().to_numpy())

    print("The morphology feature is added to adata.obsm['X_morphology']!")

    return adata if copy else None



