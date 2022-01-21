import scanpy as sc
from .._settings import settings
from pathlib import Path
from anndata import AnnData


def example_bcba() -> AnnData:
    """\
    Download processed BCBA data (10X genomics published data).
    Reference: https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1
    """
    settings.datasetdir.mkdir(exist_ok=True)
    filename = settings.datasetdir / "example_bcba.h5"
    url = "https://uc3d57a8599f02aa11c7a99072ce.dl.dropboxusercontent.com/cd/0/get/BeJMbXOnccg58fDHHwX87sW1CUdl0kq2oCziBL-akIXM0ghZH5zC7VIbPM5QW-yJZfI1NXNVOZ5Ki9rJcf2AexRuqWo2XEDfDoBOv6cxh82p71eVpGdfSnh0ZIaa_nKfTFWdyvXcHlGng-6i364ylnAi/file#"
    if not filename.is_file():
        sc.readwrite._download(url=url, path=filename)
    adata = sc.read_h5ad(filename)
    return adata
