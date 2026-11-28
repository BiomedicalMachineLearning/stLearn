from typing import Any, Literal

import scanpy
from anndata import AnnData
from numpy.random.mtrand import RandomState


def leiden(
    adata: AnnData,
    resolution: float | None = None,
    *,
    random_state: int | RandomState | None = 0,
    directed: bool = False,
    use_weights: bool = False,
    flavor: Literal["leidenalg", "igraph"] = "igraph",
    n_iterations: int = 2,
    **kwargs: Any,
) -> AnnData | None:
    """\
    Wrap function scanpy.tl.leiden

    This requires having ran :func:`~scanpy.pp.neighbors` or
    :func:`~scanpy.external.pp.bbknn` first.

    Parameters
    ----------
    adata:
        The annotated data matrix.
    resolution:
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
        Set to `None` if overriding `partition_type`
        to one that doesn't accept a `resolution_parameter`.
    random_state:
        Change the initialization of the optimization.
    directed:
        Interpret the neighborhood graph as directed?
    use_weights:
        Use weights from knn graph.
    flavor:
        Which package's implementation to use.
    n_iterations:
        How many iterations of the Leiden clustering algorithm to perform.
    **kwargs:
        Any other keyword argument is passed to :func:`scanpy.tl.leiden`, for
        example `restrict_to`, `partition_type`, `use` for the graph to
        cluster or `out` for where to write the labels.

    Returns
    -------
    None
        Updates ``adata`` with the following fields:

        - ``adata.obs['leiden']`` (:class:`pandas.Series`, dtype ``category``) -
          Array of dim (number of samples) that stores the subgroup id
          (``'0'``, ``'1'``, ...) for each cell.
    """

    if flavor == "igraph" and directed:
        raise ValueError(
            "directed=True is not supported with flavor='igraph'. "
            "Either set directed=False or use flavor='leidenalg'."
        )

    print("Applying Leiden cluster ...")
    adata = scanpy.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        directed=directed,
        use_weights=use_weights,
        flavor=flavor,
        n_iterations=n_iterations,
        **kwargs,
    )

    print("Leiden cluster is done! The labels are stored in adata.obs['leiden']")

    return adata
