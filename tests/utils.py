import os

import numpy as np
import scanpy as sc
from PIL import Image


def path_for_tests():
    return os.path.dirname(os.path.realpath(__file__))


def path_for_test_data():
    return f"{path_for_tests()}/test_data"


def read_test_data():
    """Reads in test data to run unit tests."""
    adata = sc.read_h5ad(f"{path_for_test_data()}/test_data.h5")
    im = Image.open(f"{path_for_test_data()}/test_image.jpg")
    adata.uns["spatial"]["V1_Breast_Cancer_Block_A_Section_1"]["images"]["hires"] = (
        np.array(im)
    )
    return adata
