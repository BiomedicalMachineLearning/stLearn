import numpy as np
import scanpy
from anndata import AnnData
from numpy.random.mtrand import RandomState

from .._compat import Literal

_InitPos = Literal["paga", "spectral", "random"]


def run_umap(
    adata: AnnData,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: _InitPos | np.ndarray | None = "spectral",
    random_state: int | RandomState | None = 0,
    a: float | None = None,
    b: float | None = None,
    copy: bool = False,
    method: Literal["umap", "rapids"] = "umap",  # noqa: F821
) -> AnnData | None:
    """\
    Wrap function scanpy.pp.umap
    Embed the neighborhood graph using UMAP [McInnes18]_.
    UMAP (Uniform Manifold Approximation and Projection) is a manifold learning
    technique suitable for visualizing high-dimensional data. Besides tending to
    be faster than tSNE, it optimizes the embedding such that it best reflects
    the topology of the data, which we represent throughout Scanpy using a
    neighborhood graph. tSNE, by contrast, optimizes the distribution of
    nearest-neighbor distances in the embedding such that these best match the
    distribution of distances in the high-dimensional space.  We use the
    implementation of `umap-learn <https://github.com/lmcinnes/umap>`__
    [McInnes18]_. For a few comparisons of UMAP with tSNE, see this `preprint
    <https://doi.org/10.1101/298430>`__.
    Parameters
    ----------
    adata
        Annotated data matrix.
    n_components
        The number of dimensions of the embedding.
    random_state
        If `int`, `random_state` is the seed used by the random number generator;
        If `RandomState`, `random_state` is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.
    `X_umap` : :class:`numpy.ndarray` (`adata.obsm`)
        Independent Component Analysis representation of data.

    """

    adata = scanpy.tl.umap(
        adata,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        maxiter=maxiter,
        alpha=alpha,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        init_pos=init_pos,
        random_state=random_state,
        a=a,
        b=b,
        copy=copy,
        method=method,
    )

    print("UMAP is done! Generated in adata.obsm['X_umap'] nad adata.uns['umap']")

    return adata
