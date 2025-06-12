from typing import Literal

import numpy as np
from anndata import AnnData
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

from .model_zoo import Model

_CNN_BASE = Literal["resnet50", "vgg16", "inception_v3", "xception"]


def extract_feature(
    adata: AnnData,
    cnn_base: _CNN_BASE = "resnet50",
    n_components: int = 50,
    seeds: int = 1,
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
    verbose:
        Verbose output
    copy:
        Return a copy instead of writing to adata.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    **X_morphology** : `adata.obsm` field
        Dimension reduced latent morphological features.
    Raises
    ------
    ValueError
        If any image fails to process or if tile_path column is missing.
    """

    adata = adata.copy() if copy else adata

    if "tile_path" not in adata.obs:
        raise ValueError("Please run the function stlearn.pp.tiling")

    model = Model(cnn_base)

    # Pre-allocate feature matrix, spot names and arrays to avoid overhead
    tile_paths = adata.obs["tile_path"].values
    n_spots = len(tile_paths)
    if n_spots == 0:
        raise ValueError("No tile paths found in adata.obs['tile_path']")

    first_features = _read_and_predict(tile_paths[0], model, verbose=verbose)
    n_features = len(first_features)

    # Setup feature matrix
    feature_matrix = np.empty((n_spots, n_features), dtype=np.float32)
    feature_matrix[0] = first_features

    with tqdm(
        total=n_spots,
        desc="Extract feature",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
        initial=1,  # We already processed the first image
    ) as pbar:
        for i in range(1, n_spots):
            features = _read_and_predict(tile_paths[i], model, verbose=verbose)
            feature_matrix[i] = features
            if i % 100 == 0:
                pbar.update(100)

    adata.obsm["X_tile_feature"] = feature_matrix
    pca = PCA(n_components=n_components, random_state=seeds)
    pca.fit(feature_matrix)
    adata.obsm["X_morphology"] = pca.transform(feature_matrix)

    print("The morphology feature is added to adata.obsm['X_morphology']!")

    return adata if copy else None


def _read_and_predict(path, model, verbose=False):
    try:
        with Image.open(path) as img:
            tile = np.asarray(img, dtype=np.float32)

        if verbose:
            print(f"Loaded image: {path}")

        tile = tile[np.newaxis, ...]
        return model.predict(tile).ravel()
    except Exception as e:
        raise ValueError(f"Failed to process image: {path}. Error: {str(e)}")
