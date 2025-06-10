from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from PIL import Image

# Test progress bar
from tqdm import tqdm

from .model_zoo import Model, encode
from sklearn.decomposition import PCA

_CNN_BASE = Literal["resnet50", "vgg16", "inception_v3", "xception"]

def extract_feature(
    adata: AnnData,
    cnn_base: _CNN_BASE = "resnet50",
    n_components: int = 50,
    seeds: int = 1,
    batch_size: int = 32,
    verbose: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """\
    Extract latent morphological features from H&E images using pre-trained
    convolutional neural network base

    Parameters
    ----------
    adata:
        Annotated data matrix.
    cnn_base:
        Established convolutional neural network bases
        choose one from ['resnet50', 'vgg16', 'inception_v3', 'xception']
    n_components:
        Number of principal components to compute for latent morphological features
    seeds:
        Fix random state
    batch_size:
        Number of images to process in each batch (default: 32)
    verbose:
        Verbose output
    copy:
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **X_morphology** : `adata.obsm` field
        Dimension reduced latent morphological features.
    """

    adata = adata.copy() if copy else adata

    if "tile_path" not in adata.obs:
        raise ValueError("Please run the function stlearn.pp.tiling")

    model = Model(cnn_base)
    n_spots = len(adata)
    spots = list(adata.obs["tile_path"].items())

    spot_names = []
    feature_matrix = None
    current_row = 0

    with tqdm(
        total=n_spots,
        desc="Extract feature",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:

        for i in range(0, n_spots, batch_size):
            batch_spots = spots[i:i + batch_size]
            batch_tiles, batch_spot_names = _load_batch_images(batch_spots, verbose)

            if batch_tiles:
                batch_array = np.stack(batch_tiles, axis=0)
                batch_features = model.predict(batch_array)

                if feature_matrix is None:
                    n_features = batch_features.shape[1]
                    feature_matrix = np.empty((n_spots, n_features),
                                              dtype=np.float32)

                end_row = current_row + len(batch_features)
                feature_matrix[current_row:end_row] = batch_features
                current_row = end_row

                spot_names.extend(batch_spot_names)

            pbar.update(len(batch_spots))

    if feature_matrix is None or current_row == 0:
        raise ValueError("No features were successfully extracted")

    feature_matrix = feature_matrix[:current_row]

    feature_df = pd.DataFrame(feature_matrix.T, columns=spot_names)
    feature_array = feature_df.T.to_numpy()

    adata.obsm["X_tile_feature"] = feature_array

    pca = PCA(n_components=n_components, random_state=seeds)
    adata.obsm["X_morphology"] = pca.fit_transform(feature_matrix)

    print("The morphology feature is added to adata.obsm['X_morphology']!")

    return adata if copy else None


def _load_batch_images(batch_spots, verbose=False):
    """Load a batch of images from file paths."""
    images = []
    names = []

    for spot_name, tile_path in batch_spots:
        try:
            image = np.asarray(Image.open(tile_path), dtype=np.float32)
            images.append(image)
            names.append(spot_name)

            if verbose:
                print(f"Loaded image for spot: {spot_name}")

        except Exception as e:
            print(f"Warning: Failed to load image for spot {spot_name}: {e}")
            continue

    return images, names
