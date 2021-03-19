import scanpy as sc
from PIL import Image
import numpy as np

def read_test_data():
	adata = sc.read_h5ad("./tests/test_data/test_data.h5")
	im = Image.open("./tests/test_data/test_image.jpg")
	adata.uns["spatial"]["V1_Breast_Cancer_Block_A_Section_1"]["images"]["hires"] = np.array(im)
	return adata


