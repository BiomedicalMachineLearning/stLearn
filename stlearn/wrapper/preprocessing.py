from anndata import AnnData
import stlearn
import os
import scanpy

def Preprocessing(
    adata: AnnData,
    out_path: str = "../tiling",
    crop_size: float = 40,
    use_data: str = "X_pca",
    radius: float = 50,
    adjust_method: str = "mean",
    n_neighbors: int = 25,
    use_label: str = "louvain",
    n_clusters: int = 10,
    random_state: int = 0,
    copy: bool = False,

    ) -> AnnData:

    scanpy.pp.filter_genes()
    scanpy.pp.filter_cells()
    scanpy.pp.normalize_per_cell(adata, counts_per_cell_after=1e6)
    scanpy.pp.log1p(adata)

    return adata if copy else None
