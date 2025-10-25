from typing import Literal

import numpy as np
from PIL import Image
from anndata import AnnData
from sklearn.decomposition import PCA
from tqdm import tqdm

from .model_zoo import Model

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

    # Load model
    if verbose:
        print(f"Loading {cnn_base} model...")
    model = Model(cnn_base, batch_size=batch_size)

    # Pre-allocate feature matrix, spot names and arrays to avoid overhead
    tile_paths = adata.obs["tile_path"].values
    n_spots = len(tile_paths)
    if n_spots == 0:
        raise ValueError("No tile paths found in adata.obs['tile_path']")

    # Load tiled images.
    if verbose:
        print(f"Processing {n_spots} image tiles...")
    images = []
    for path in tqdm(tile_paths, desc="Loading tiles", disable=not verbose):
        try:
            with Image.open(path) as img:
                tile = np.asarray(img, dtype=np.float32)
            images.append(tile)
        except Exception as e:
            raise ValueError(f"Failed to load image: {path}. Error: {e}")

    images = np.array(images, dtype=np.float32)

    # Get features
    all_features = []
    for i in tqdm(range(0, n_spots, batch_size), desc="Predicting",
                  disable=not verbose):
        batch = images[i:i + batch_size]
        features = model.predict(batch)
        all_features.append(features)

    # Save matrix
    feature_matrix = np.vstack(all_features)
    adata.obsm["X_tile_feature"] = feature_matrix

    if verbose:
        print("Running PCA dimensionality reduction...")
    pca = PCA(n_components=n_components, random_state=seeds)
    pca.fit(feature_matrix)
    if verbose:
        print(
            f"PCA complete! Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    adata.obsm["X_morphology"] = pca.transform(feature_matrix)

    if verbose:
        print("The morphology feature is added to adata.obsm['X_morphology']!")

    return adata if copy else None
