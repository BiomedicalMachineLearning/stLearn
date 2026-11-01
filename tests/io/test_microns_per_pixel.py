"""microns_per_pixel from the readers: set for Visium from the 55 µm spot, kept
from Space Ranger for Visium HD, and absent (an error) when nothing records it."""

import json
import pathlib
import tempfile
import unittest

import anndata
import h5py
import numpy as np
import pandas as pd
import PIL.Image

import stlearn as st
from stlearn.io.spatialdata import visium_scalefactors

# Standard Visium: a 55 µm spot is 200 full-resolution pixels, so 0.275 µm each.
VISIUM = {
    "spot_diameter_fullres": 200.0,
    "fiducial_diameter_fullres": 300.0,
    "tissue_hires_scalef": 0.1,
    "tissue_lowres_scalef": 0.03,
}
# Visium HD: Space Ranger gives microns_per_pixel itself.
VISIUM_HD = dict(VISIUM, microns_per_pixel=0.2738)

BARCODES = ["AAAC-1", "AAAG-1", "AACC-1", "AAGG-1"]
FULLRES_XY = np.array([[1000.0, 1000.0], [1400.0, 1000.0], [1200.0, 1350.0], [0, 0]])


def write_spaceranger(path: pathlib.Path, scalefactors: dict) -> None:
    """A minimal Space Ranger output folder: counts, positions, scale factors and
    images."""
    genes = ["GeneA", "GeneB", "GeneC"]
    counts = np.arange(1, len(BARCODES) * len(genes) + 1).reshape(len(genes), -1)
    with h5py.File(path / "filtered_feature_bc_matrix.h5", "w") as f:
        f.attrs["library_ids"] = np.array([b"lib"])
        matrix = f.create_group("matrix")
        # Compressed sparse columns: one column per barcode, as Cell Ranger writes.
        nonzero = counts.T
        matrix["data"] = nonzero.ravel().astype(np.int32)
        matrix["indices"] = np.tile(np.arange(len(genes)), len(BARCODES))
        matrix["indptr"] = np.arange(0, nonzero.size + 1, len(genes))
        matrix["shape"] = np.array([len(genes), len(BARCODES)])
        matrix["barcodes"] = np.array(BARCODES, dtype="S")
        features = matrix.create_group("features")
        features["id"] = np.array([f"ENSG{i}" for i in range(len(genes))], dtype="S")
        features["name"] = np.array(genes, dtype="S")
        features["feature_type"] = np.array(["Gene Expression"] * 3, dtype="S")
        features["genome"] = np.array(["GRCh38"] * 3, dtype="S")

    spatial = path / "spatial"
    spatial.mkdir()
    pd.DataFrame(
        {
            "barcode": BARCODES,
            "in_tissue": 1,
            "array_row": [0, 0, 1, 2],
            "array_col": [0, 2, 1, 4],
            "pxl_row_in_fullres": FULLRES_XY[:, 1].astype(int),
            "pxl_col_in_fullres": FULLRES_XY[:, 0].astype(int),
        }
    ).to_csv(spatial / "tissue_positions.csv", index=False)
    (spatial / "scalefactors_json.json").write_text(json.dumps(scalefactors))
    for name, size in (("hires", 200), ("lowres", 60)):
        PIL.Image.new("RGB", (size, size), "white").save(
            spatial / f"tissue_{name}_image.png"
        )


def spots() -> anndata.AnnData:
    adata = anndata.AnnData(np.ones((len(FULLRES_XY), 1), dtype=np.float32))
    adata.obs_names = BARCODES
    adata.obsm["spatial"] = FULLRES_XY.copy()
    return adata


class TestMicronsPerPixel(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_from_spot_diameter(self):
        self.assertAlmostEqual(visium_scalefactors(VISIUM)["microns_per_pixel"], 0.275)

    def test_keeps_space_ranger_value(self):
        self.assertEqual(visium_scalefactors(VISIUM_HD)["microns_per_pixel"], 0.2738)

    def test_no_spot_diameter(self):
        """Nothing to work it out from, so it stays absent."""
        scalefactors = {}
        self.assertNotIn("microns_per_pixel", visium_scalefactors(scalefactors))

    def test_does_not_change_input(self):
        scalefactors = dict(VISIUM)
        visium_scalefactors(scalefactors)
        self.assertEqual(scalefactors, VISIUM)

    def test_read_10x(self):
        """A 55 µm spot is spot_diameter_fullres times tissue_hires_scalef pixels."""
        write_spaceranger(self.path, VISIUM)
        adata = st.io.read_10x(self.path)
        scalefactors = adata.uns["spatial"]["lib"]["scalefactors"]
        self.assertAlmostEqual(scalefactors["microns_per_pixel"], 55 / 200)
        self.assertAlmostEqual(st.utils.microns_to_pixels(adata, 55), 200 * 0.1)

    def test_attach_spatial_visium(self):
        write_spaceranger(self.path, VISIUM)
        adata = st.io.attach_spatial(spots(), self.path / "spatial", "lib")
        scalefactors = adata.uns["spatial"]["lib"]["scalefactors"]
        self.assertAlmostEqual(scalefactors["microns_per_pixel"], 55 / 200)
        self.assertAlmostEqual(st.utils.microns_to_pixels(adata, 55), 200 * 0.1)

    def test_attach_spatial_visium_hd(self):
        write_spaceranger(self.path, VISIUM_HD)
        adata = st.io.attach_spatial(spots(), self.path / "spatial", "lib")
        scalefactors = adata.uns["spatial"]["lib"]["scalefactors"]
        self.assertEqual(scalefactors["microns_per_pixel"], 0.2738)

    def test_no_physical_scale(self):
        """create_stlearn's coordinates have no recorded scale, so converting to
        microns is an error naming the library and how to set it."""
        count = pd.DataFrame(np.ones((3, 2)), index=["c0", "c1", "c2"])
        count.columns = ["GeneA", "GeneB"]
        spatial = pd.DataFrame(
            {"imagecol": [0.0, 10.0, 20.0], "imagerow": [0.0, 5.0, 10.0]},
            index=count.index,
        )
        adata = st.io.create_stlearn(count, spatial, "slide")
        with self.assertRaisesRegex(
            ValueError, r"microns_per_pixel missing in 'slide'"
        ):
            st.utils.microns_to_pixels(adata, 55)
        with self.assertRaisesRegex(ValueError, "microns_per_pixel"):
            st.tl.cci.create_grid(adata, verbose=False)


if __name__ == "__main__":
    unittest.main()
