import os
import scanpy as sc
from PIL import Image
import numpy as np


def read_test_data():
    """Reads in test data to run unit tests."""
    # Determining path of this file #
    path = os.path.dirname(os.path.realpath(__file__))
    adata = sc.read_h5ad(f"{path}/test_data/test_data.h5")
    im = Image.open(f"{path}/test_data/test_image.jpg")
    adata.uns["spatial"]["V1_Breast_Cancer_Block_A_Section_1"]["images"][
        "hires"
    ] = np.array(im)
    return adata
