# !/usr/bin/env python

"""Tests for Model."""

import unittest

import numpy as np

from stlearn.image_preprocessing.model_zoo import Model


class TestModelZoo(unittest.TestCase):
    def _make_input(self, n=5, size=224):
        return np.random.Generator(0, 256, size=(n, size, size, 3), dtype=np.uint8)

    def test_resnet50_v1_output_shape(self):
        model = Model(base="resnet50", batch_size=4)
        out = model.predict(self._make_input())
        self.assertEqual(out.shape, (5, 2048))
        self.assertEqual(out.dtype, np.float32)

    def test_resnet50_v2_output_shape(self):
        model = Model(base="resnet50", weights="v2", batch_size=4)
        out = model.predict(self._make_input())
        self.assertEqual(out.shape, (5, 2048))

    def test_batching_works(self):
        """Inputs larger than batch_size should be chunked transparently."""
        model = Model(base="resnet50", batch_size=4)
        # 10 > batch_size=4, exercises chunking
        x = self._make_input(n=10)
        out = model.predict(x)
        self.assertEqual(out.shape, (10, 2048))

    def test_invalid_base_raises(self):
        with self.assertRaisesRegex(ValueError, "not a valid model"):
            Model(base="not_a_real_backbone")

    def test_invalid_weights_version_raises(self):
        with self.assertRaisesRegex(ValueError, "Available versions"):
            Model(base="vgg16", weights="v2")

    def test_predict_rejects_wrong_ndim(self):
        model = Model(base="resnet50", batch_size=4)
        with self.assertRaisesRegex(ValueError, "expected NHWC array"):
            model.predict(np.zeros((224, 224, 3), dtype=np.uint8))  # 3D, not 4D

    def test_predict_rejects_wrong_channels(self):
        model = Model(base="resnet50", batch_size=4)
        x = np.zeros((5, 224, 224, 4), dtype=np.uint8)  # RGBA
        with self.assertRaisesRegex(ValueError, "3 channels"):
            model.predict(x)
