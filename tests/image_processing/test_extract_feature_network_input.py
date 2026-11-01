import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from anndata import AnnData
from PIL import Image

import stlearn as st
from stlearn.image_preprocessing import feature_extractor
from stlearn.image_preprocessing.model_zoo import Model


class NetworkInputRecorder(Model):
    """The real Model, but records the largest value it feeds the network."""

    largest_input = 0.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        preprocess = self.preprocess

        def recording_preprocess(img):
            tensor = preprocess(img)
            NetworkInputRecorder.largest_input = max(
                NetworkInputRecorder.largest_input, float(tensor.abs().max())
            )
            return tensor

        self.preprocess = recording_preprocess


class TestExtractFeatureNetworkInput(unittest.TestCase):
    def test_network_input_is_imagenet_normalised(self):
        """Tiles must reach the network scaled to [0, 1] and then normalised.

        ImageNet normalisation keeps [0, 1] pixels within about +/-2.64.
        Tiles read as 0-255 floats skip the scaling and reach about 1136.
        """
        NetworkInputRecorder.largest_input = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, colour in enumerate([(0, 0, 0), (128, 128, 128), (255, 255, 255)]):
                path = Path(tmp) / f"tile_{i}.png"
                Image.new("RGB", (64, 64), colour).save(path)
                paths.append(str(path))
            adata = AnnData(np.zeros((len(paths), 1), dtype=np.float32))
            adata.obs["tile_path"] = paths

            with mock.patch.object(feature_extractor, "Model", NetworkInputRecorder):
                st.pp.extract_feature(adata, n_components=2)

        self.assertGreater(NetworkInputRecorder.largest_input, 0.0)
        self.assertLess(NetworkInputRecorder.largest_input, 3.0)
