# from .embedding.scvi import run_ldvae
from .embedding.diffmap import run_diffmap

# from .embedding.scvi import run_ldvae
from .embedding.fa import run_fa
from .embedding.ica import run_ica
from .embedding.pca import run_pca
from .embedding.umap import run_umap

__all__ = [
    "run_pca",
    "run_umap",
    "run_ica",
    "run_fa",
    "run_diffmap",
]
