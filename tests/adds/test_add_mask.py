#!/usr/bin/env python

"""Tests for add_mask, apply_mask and plot_mask."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import stlearn as st
from tests.utils import read_test_data


class TestMask(unittest.TestCase):
    """A mask covering the top half of the image annotates the spots under it."""

    @classmethod
    def setUpClass(cls):
        cls._base_adata = read_test_data()
        library_id = next(iter(cls._base_adata.uns["spatial"]))
        image = cls._base_adata.uns["spatial"][library_id]["images"]["hires"]
        cls._tmp = tempfile.TemporaryDirectory()
        cls._mask_path = Path(cls._tmp.name) / "mask.png"
        # Black selects (the default), so the top half is black and the rest white.
        mask = np.full(image.shape, 255, dtype=np.uint8)
        cls._split_row = 1300
        mask[: cls._split_row] = 0
        Image.fromarray(mask).save(cls._mask_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.adata = self.__class__._base_adata.copy()
        st.add.add_mask(self.adata, imgpath=self._mask_path, key="tumour")

    def tearDown(self):
        plt.close("all")

    def test_add_mask(self):
        self.assertIn("mask_image", self.adata.uns)

    def test_apply_mask_annotates_spots(self):
        st.add.apply_mask(self.adata, masks="all")

        inside = self.adata.obs["imagerow"] < self._split_row
        self.assertGreater(inside.sum(), 0)
        self.assertGreater((~inside).sum(), 0)
        annotation = self.adata.obs["mask_annotation"]
        self.assertTrue((annotation[inside] == "tumour").all())
        self.assertTrue((annotation[~inside] == "unassigned").all())
        self.assertTrue((self.adata.obs["mask_annotation_code"][inside] == 0).all())
        self.assertTrue((self.adata.obs["mask_annotation_code"][~inside] == -1).all())

    def test_apply_mask_palettes(self):
        for cmap_name in ("default", "vega_10_scanpy", "jana_40"):
            with self.subTest(cmap_name=cmap_name):
                st.add.apply_mask(self.adata, masks="all", cmap_name=cmap_name)
                self.assertIn("mask_annotation", self.adata.uns)

    def test_plot_mask(self):
        st.add.apply_mask(self.adata, masks="all")
        with patch("matplotlib.pyplot.show"):
            st.pl.plot_mask(self.adata)
