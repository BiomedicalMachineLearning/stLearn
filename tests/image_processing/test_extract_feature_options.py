import tempfile
import unittest
from pathlib import Path
from typing import get_args, get_type_hints
from unittest import mock

import numpy as np
from anndata import AnnData
from PIL import Image
from torchvision import models

from stlearn.image_preprocessing import feature_extractor


class TestExtractFeatureOptions(unittest.TestCase):
    def test_every_cnn_base_has_v1_weights(self):
        hints = get_type_hints(feature_extractor.extract_feature)
        for base in get_args(hints["cnn_base"]):
            with self.subTest(base):
                self.assertIsNotNone(models.get_model_weights(base).IMAGENET1K_V1)

    def test_unsupported_cnn_base_lists_valid_choices(self):
        adata = AnnData(np.zeros((1, 1), dtype=np.float32))
        adata.obs["tile_path"] = ["unused.png"]
        with self.assertRaisesRegex(ValueError, "xception.*convnext_tiny"):
            feature_extractor.extract_feature(adata, cnn_base="xception")

    def test_weights_passed_to_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, grey in enumerate([0, 255]):
                path = Path(tmp) / f"tile_{i}.png"
                Image.new("RGB", (8, 8), (grey, grey, grey)).save(path)
                paths.append(str(path))
            adata = AnnData(np.zeros((len(paths), 1), dtype=np.float32))
            adata.obs["tile_path"] = paths

            with mock.patch.object(feature_extractor, "Model") as model_class:
                model_class.return_value.predict.side_effect = lambda x: np.full(
                    (len(x), 4), x.mean(), dtype=np.float32
                )
                feature_extractor.extract_feature(adata, weights="v2", n_components=1)

        self.assertEqual(model_class.call_args.kwargs["weights"], "v2")
