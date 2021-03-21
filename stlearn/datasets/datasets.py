import scanpy as sc
from .._settings import settings
from pathlib import Path
from anndata import AnnData


def example_bcba() -> AnnData:
    """\
    Download processed BCBA data (10X genomics published data).
    Reference: https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1
    """
    filename = settings.datasetdir / "example_bcba.h5"
    url = "https://srv-store5.gofile.io/download/4oQojj/3377e2ef96224abcf80a1bf279eeced6/example_bcba_small.h5"
    if not filename.is_file():
        sc.readwrite._download(url=url, path=filename)
    adata = sc.read_h5ad(filename)
    return adata
