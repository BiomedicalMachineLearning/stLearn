import os

import numpy as np
import scanpy as sc
from PIL import Image


def test_path():
    return os.path.dirname(os.path.realpath(__file__))


def test_data_path():
    return f"{test_path()}/test_data"


def read_test_data():
    """Reads in test data to run unit tests."""
    adata = sc.read_h5ad(f"{test_data_path()}/test_data.h5")
    im = Image.open(f"{test_data_path()}/test_image.jpg")
    adata.uns["spatial"]["V1_Breast_Cancer_Block_A_Section_1"]["images"]["hires"] = (
        np.array(im)
    )
    return adata
