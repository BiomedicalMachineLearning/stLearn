from numpy import linalg as la


def lambda_dist(A1, A2, k=None, p=2, kind="laplacian"):
    """The function is migrated from NetComp package. The lambda distance between graphs, which is defined as
        d(G1,G2) = norm(L_1 - L_2)
    where L_1 is a vector of the top k eigenvalues of the appropriate matrix
    associated with G1, and L2 is defined similarly.
    Parameters
    ----------
    A1, A2 : NumPy matrices
        Adjacency matrices of graphs to be compared
    k : Integer
        The number of eigenvalues to be compared
    p : non-zero Float
        The p-norm is used to compare the resulting vector of eigenvalues.

    kind : String , in {'laplacian','laplacian_norm','adjacency'}
        The matrix for which eigenvalues will be calculated.
    Returns
    -------
    dist : float
        The distance between the two graphs
    Notes
    -----
    The norm can be any p-norm; by default we use p=2. If p<0 is used, the
    result is not a mathematical norm, but may still be interesting and/or
    useful.
    If k is provided, then we use the k SMALLEST eigenvalues for the Laplacian
    distances, and we use the k LARGEST eigenvalues for the adjacency
    distance. This is because the corresponding order flips, as L = D-A.
    References
    ----------
    See Also
    --------
    netcomp.linalg._eigs
    normalized_laplacian_eigs
    """
    # ensure valid k
    n1, n2 = [A.shape[0] for A in [A1, A2]]
    N = min(n1, n2)  # minimum size between the two graphs
    if k is None or k > N:
        k = N

    # form matrices
    L1, L2 = [laplacian_matrix(A) for A in [A1, A2]]
    # get eigenvalues, ignore eigenvectors
    evals1, evals2 = [_eigs(L)[0] for L in [L1, L2]]

    dist = la.norm(evals1[:k] - evals2[:k], ord=p)
    return dist


def resistance_distance(
    A1, A2, p=2, renormalized=False, attributed=False, check_connected=True, beta=1
):
    """Compare two graphs using resistance distance (possibly renormalized).
    Parameters
    ----------
    A1, A2 : NumPy Matrices
        Adjacency matrices of graphs to be compared.
    p : float
        Function returns the p-norm of the flattened matrices.
    renormalized : Boolean, optional (default = False)
        If true, then renormalized resistance distance is computed.
    attributed : Boolean, optional (default=False)
        If true, then the resistance distance PER NODE is returned.
    check_connected : Boolean, optional (default=True)
        If false, then no check on connectivity is performed. See Notes of
        resistance_matrix for more information.
    beta : float, optional (default=1)
        A parameter used in the calculation of the renormalized resistance
        matrix. If using regular resistance, this is irrelevant.
    Returns
    -------
    dist : float of numpy array
        The RR distance between the two graphs. If attributed is True, then
        vector distance per node is returned.
    Notes
    -----
    The distance is calculated by assuming the nodes are in correspondence, and
    any nodes not present are treated as isolated by renormalized resistance.
    References
    ----------
    See Also
    --------
    resistance_matrix
    """
    # Calculate resistance matricies and compare
    if renormalized:
        # pad smaller adj. mat. so they're the same size
        n1, n2 = [A.shape[0] for A in [A1, A2]]
        N = max(n1, n2)
        A1, A2 = [_pad(A, N) for A in [A1, A2]]
        R1, R2 = [renormalized_res_mat(A, beta=beta) for A in [A1, A2]]
    else:
        R1, R2 = [
            resistance_matrix(A, check_connected=check_connected) for A in [A1, A2]
        ]
    try:
        distance_vector = np.sum((R1 - R2) ** p, axis=1)
    except ValueError:
        raise InputError(
            "Input matrices are different sizes. Please use "
            "renormalized resistance distance."
        )
    if attributed:
        return distance_vector ** (1 / p)
    else:
        return np.sum(distance_vector) ** (1 / p)


# **********
# Eigenstuff
# **********
# Functions for calculating eigenstuff of graphs.


from scipy import sparse as sps
import numpy as np
from scipy.sparse import linalg as spla
from numpy import linalg as la

from scipy.sparse import issparse

######################
## Helper Functions ##
######################


def _eigs(M, which="SR", k=None):
    """Helper function for getting eigenstuff.
    Parameters
    ----------
    M : matrix, numpy or scipy sparse
        The matrix for which we hope to get eigenstuff.
    which : string in {'SR','LR'}
        If 'SR', get eigenvalues with smallest real part. If 'LR', get largest.
    k : int
        Number of eigenvalues to return
    Returns
    -------
    evals, evecs : numpy arrays
        Eigenvalues and eigenvectors of matrix M, sorted in ascending or
        descending order, depending on 'which'.
    See Also
    --------
    numpy.linalg.eig
    scipy.sparse.eigs
    """
    n, _ = M.shape
    if k is None:
        k = n
    if which not in ["LR", "SR"]:
        raise ValueError("which must be either 'LR' or 'SR'.")
    M = M.astype(float)
    if issparse(M) and k < n - 1:
        evals, evecs = spla.eigs(M, k=k, which=which)
    else:
        try:
            M = M.todense()
        except:
            pass
        evals, evecs = la.eig(M)
        # sort dem eigenvalues
        inds = np.argsort(evals)
        if which == "LR":
            inds = inds[::-1]
        else:
            pass
        inds = inds[:k]
        evals = evals[inds]
        evecs = np.array(evecs[:, inds])
    return np.real(evals), np.real(evecs)


#####################
##  Get Eigenstuff ##
#####################


def normalized_laplacian_eig(A, k=None):
    """Return the eigenstuff of the normalized Laplacian matrix of graph
    associated with adjacency matrix A.
    Calculates via eigenvalues if
    K = D^(-1/2) A D^(-1/2)
    where `A` is the adjacency matrix and `D` is the diagonal matrix of
    node degrees. Since L = I - K, the eigenvalues and vectors of L can
    be easily recovered.
    Parameters
    ----------
    A : NumPy matrix
        Adjacency matrix of a graph
    k : int, 0 < k < A.shape[0]-1
        The number of eigenvalues to grab.
    Returns
    -------
    lap_evals : NumPy array
       Eigenvalues of L
    evecs : NumPy matrix
       Columns are the eigenvectors of L
    Notes
    -----
    This way of calculating the eigenvalues of the normalized graph laplacian is
    more numerically stable than simply forming the matrix L = I - K and doing
    numpy.linalg.eig on the result. This is because the eigenvalues of L are
    close to zero, whereas the eigenvalues of K are close to 1.
    References
    ----------
    See Also
    --------
    nx.laplacian_matrix
    nx.normalized_laplacian_matrix
    """
    n, m = A.shape
    ##
    ## TODO: implement checks on the adjacency matrix
    ##
    degs = _flat(A.sum(axis=1))
    # the below will break if
    inv_root_degs = [d ** (-1 / 2) if d > _eps else 0 for d in degs]
    inv_rootD = sps.spdiags(inv_root_degs, [0], n, n, format="csr")
    # build normalized diffusion matrix
    K = inv_rootD * A * inv_rootD
    evals, evecs = _eigs(K, k=k, which="LR")
    lap_evals = 1 - evals
    return np.real(lap_evals), np.real(evecs)


#     """
# ********
# Matrices
# ********
# Matrices associated with graphs. Also contains linear algebraic helper functions.
# """


from scipy import sparse as sps
from scipy.sparse import issparse
import numpy as np

_eps = 10 ** (-10)  # a small parameter

######################
## Helper Functions ##
######################


def _flat(D):
    """Flatten column or row matrices, as well as arrays."""
    if issparse(D):
        raise ValueError("Cannot flatten sparse matrix.")
    d_flat = np.array(D).flatten()
    return d_flat


def _pad(A, N):
    """Pad A so A.shape is (N,N)"""
    n, _ = A.shape
    if n >= N:
        return A
    else:
        if issparse(A):
            # thrown if we try to np.concatenate sparse matrices
            side = sps.csr_matrix((n, N - n))
            bottom = sps.csr_matrix((N - n, N))
            A_pad = sps.hstack([A, side])
            A_pad = sps.vstack([A_pad, bottom])
        else:
            side = np.zeros((n, N - n))
            bottom = np.zeros((N - n, N))
            A_pad = np.concatenate([A, side], axis=1)
            A_pad = np.concatenate([A_pad, bottom])
        return A_pad


########################
## Matrices of Graphs ##
########################


def degree_matrix(A):
    """Diagonal degree matrix of graph with adjacency matrix A
    Parameters
    ----------
    A : matrix
        Adjacency matrix
    Returns
    -------
    D : SciPy sparse matrix
        Diagonal matrix of degrees.
    """
    n, m = A.shape
    degs = _flat(A.sum(axis=1))
    D = sps.spdiags(degs, [0], n, n, format="csr")
    return D


def laplacian_matrix(A, normalized=False):
    """Diagonal degree matrix of graph with adjacency matrix A
    Parameters
    ----------
    A : matrix
        Adjacency matrix
    normalized : Bool, optional (default=False)
        If true, then normalized laplacian is returned.
    Returns
    -------
    L : SciPy sparse matrix
        Combinatorial laplacian matrix.
    """
    n, m = A.shape
    D = degree_matrix(A)
    L = D - A
    if normalized:
        degs = _flat(A.sum(axis=1))
        rootD = sps.spdiags(np.power(degs, -1 / 2), [0], n, n, format="csr")
        L = rootD * L * rootD
    return L


# """
# **********
# Exceptions
# **********
# Custom exceptions for NetComp.
# """


class UndefinedException(Exception):
    """Raised when matrix to be returned is undefined"""


# """
# **********
# Resistance
# **********
# Resistance matrix. Renormalized version, as well as conductance and commute matrices.
# """

import networkx as nx
from numpy import linalg as la
from scipy import linalg as spla
import numpy as np
from scipy.sparse import issparse

# from netcomp.linalg.matrices import laplacian_matrix
# from netcomp.exception import UndefinedException


def resistance_matrix(A, check_connected=True):
    """Return the resistance matrix of G.
    Parameters
    ----------
    A : NumPy matrix or SciPy sparse matrix
        Adjacency matrix of a graph.
    check_connected : Boolean, optional (default=True)
        If false, then the resistance matrix will be computed even for
        disconnected matrices. See Notes.
    Returns
    -------
    R : NumPy matrix
       Matrix of pairwise resistances between nodes.
    Notes
    -----
    Uses formula for resistance matrix R in terms of Moore-Penrose of
    pseudoinverse (non-normalized) graph Laplacian. See e.g. Theorem 2.1 in [1].
    This formula can be computed even for disconnected graphs, although the
    interpretation in this case is unclear. Thus, the usage of
    check_connected=False is recommended only to reduce computation time in a
    scenario in which the user is confident the graph in question is, in fact,
    connected.
    Since we do not expect the pseudoinverse of the laplacian to be sparse, we
    convert L to dense form before running np.linalg.pinv(). The returned
    resistance matrix is dense.
    See Also
    --------
    nx.laplacian_matrix
    References
    ----------
    .. [1] W. Ellens, et al. (2011)
       Effective graph resistance.
       Linear Algebra and its Applications, 435 (2011)
    """
    n, m = A.shape
    # check if graph is connected
    if check_connected:
        if issparse(A):
            G = nx.from_scipy_sparse_array(A)
        else:
            G = nx.from_numpy_array(A)
        if not nx.is_connected(G):
            raise UndefinedException(
                "Graph is not connected. " "Resistance matrix is undefined."
            )
    L = laplacian_matrix(A)
    try:
        L = L.todense()
    except:
        pass
    M = la.pinv(L)
    # calculate R in terms of M
    d = np.reshape(np.diag(M), (n, 1))
    ones = np.ones((n, 1))
    R = np.dot(d, ones.T) + np.dot(ones, d.T) - M - M.T
    return R


def commute_matrix(A):
    """Return the commute matrix of the graph associated with adj. matrix A.
    Parameters
    ----------
    A : NumPy matrix or SciPy sparse matrix
        Adjacency matrix of a graph.
    Returns
    -------
    C : NumPy matrix
       Matrix of pairwise resistances between nodes.
    Notes
    -----
    Uses formula for commute time matrix in terms of resistance matrix,
    C = R*2*|E|
    where |E| is the number of edges in G. See e.g. Theorem 2.8 in [1].
    See Also
    --------
    laplacian_matrix
    resistance_matrix
    References
    ----------
    .. [1] W. Ellens, et al. (2011)
       Effective graph resistance.
       Linear Algebra and its Applications, 435 (2011)
    """
    R = resistance_matrix(A)
    E = A.sum() / 2  # number of edges in graph
    C = 2 * E * R
    return C


def renormalized_res_mat(A, beta=1):
    """Return the renormalized resistance matrix of graph associated with A.
    To renormalize a resistance R, we apply the function
    R' = R / (R + beta)
    In this way, the renormalized resistance of nodes in disconnected components
    is 1. The parameter beta determines the penalty for disconnection. If we set
    beta to be approximately the maximum resistance found in the network, then
    the penalty for disconnection is at least 1/2.
    Parameters
    ----------
    A : NumPy matrix or SciPy sparse matrix
        Adjacency matrix of a graph.
    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist. If
       nodelist is None, then the ordering is produced by G.nodes().
    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.
    beta : float, optional
       Scaling parameter in renormalization. Must be greater than or equal to
       1. Determines how heavily disconnection is penalized.
    Returns
    -------
    R :  NumPy array
       Matrix of pairwise renormalized resistances between nodes.
    Notes
    -----
    This function converts to a NetworkX graph, as it uses the algorithms
    therein for identifying connected components.
    See Also
    --------
    resistance_matrix
    """
    if issparse(A):
        G = nx.from_scipy_sparse_array(A)
    else:
        G = nx.from_numpy_array(A)
    n = len(G)
    subgraphR = []
    for subgraph in nx.connected_component_subgraphs(G):
        a_sub = nx.adjacency_matrix(subgraph)
        r_sub = resistance_matrix(a_sub)
        subgraphR.append(r_sub)
    R = spla.block_diag(*subgraphR)
    # now, resort R so that it matches the original node list
    component_order = []
    for component in nx.connected_components(G):
        component_order += list(component)
    component_order = list(np.argsort(component_order))
    R = R[component_order, :]
    R = R[:, component_order]
    renorm = np.vectorize(lambda r: r / (r + beta))
    R = renorm(R)
    # set resistance for different components to 1
    R[R == 0] = 1
    R = R - np.eye(n)  # don't want diagonal to be 1
    return R


def conductance_matrix(A):
    """Return the conductance matrix of G.
    The conductance matrix of G is the element-wise inverse of the resistance
    matrix. The diagonal is set to 0, although it is formally infinite. Nodes in
    disconnected components have 0 conductance.
    Parameters
    ----------
    G : graph
       A NetworkX graph
    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist. If
       nodelist is None, then the ordering is produced by G.nodes().
    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.
    Returns
    -------
    C :  NumPy array
       Matrix of pairwise conductances between nodes.
    See Also
    --------
    resistance_matrix
    renormalized_res_mat
    """
    if issparse(A):
        G = nx.from_scipy_sparse_array(A)
    else:
        G = nx.from_numpy_array(A)
    subgraphC = []
    for subgraph in nx.connected_component_subgraphs(G):
        a_sub = nx.adjacency_matrix(subgraph)
        r_sub = resistance_matrix(a_sub)
        m = len(subgraph)
        # add one to diagonal, invert, remove one from diagonal:
        c_sub = 1 / (r_sub + np.eye(m)) - np.eye(m)
        subgraphC.append(c_sub)
    C = spla.block_diag(*subgraphC)
    # resort C so that it matches the original node list
    component_order = []
    for component in nx.connected_components(G):
        component_order += list(component)
    component_order = list(np.argsort(component_order))
    C = C[component_order, :]
    C = C[:, component_order]
    return C


########################
## CytoTrace wrapper  ##
########################

from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    TypeVar,
    Hashable,
    Iterable,
    Optional,
    Sequence,
)
import numpy as np
import pandas as pd
from pandas import Series
from scipy.stats import norm
from numpy.linalg import norm as d_norm
from scipy.sparse import eye as speye
from scipy.sparse import diags, issparse, spmatrix, csr_matrix, isspmatrix_csr
from sklearn.cluster import KMeans
from pandas.api.types import infer_dtype, is_categorical_dtype
from scipy.sparse.linalg import norm as sparse_norm


def _mat_mat_corr_sparse(
    X: csr_matrix,
    Y: np.ndarray,
) -> np.ndarray:
    """\
    This function is borrow from cellrank
    """
    n = X.shape[1]

    X_bar = np.reshape(np.array(X.mean(axis=1)), (-1, 1))
    X_std = np.reshape(
        np.sqrt(np.array(X.power(2).mean(axis=1)) - (X_bar**2)), (-1, 1)
    )

    y_bar = np.reshape(np.mean(Y, axis=0), (1, -1))
    y_std = np.reshape(np.std(Y, axis=0), (1, -1))

    with np.warnings.catch_warnings():
        np.warnings.filterwarnings(
            "ignore", r"invalid value encountered in true_divide"
        )
        return (X @ Y - (n * X_bar * y_bar)) / ((n - 1) * X_std * y_std)


def _correlation_test_helper(
    X: Union[np.ndarray, spmatrix],
    Y: np.ndarray,
    n_perms: Optional[int] = None,
    seed: Optional[int] = None,
    confidence_level: float = 0.95,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function is borrow from cellrank.
    Compute the correlation between rows in matrix ``X`` columns of matrix ``Y``.
    Parameters
    ----------
    X
        Array or matrix of `(M, N)` elements.
    Y
        Array of `(N, K)` elements.
    method
        Method for p-value calculation.
    n_perms
        Number of permutations if ``method='perm_test'``.
    seed
        Random seed if ``method='perm_test'``.
    confidence_level
        Confidence level for the confidence interval calculation. Must be in `[0, 1]`.
    kwargs
        Keyword arguments for :func:`cellrank.ul._parallelize.parallelize`.
    Returns
    -------
        Correlations, p-values, corrected p-values, lower and upper bound of 95% confidence interval.
        Each array if of shape ``(n_genes, n_lineages)``.
    """

    def perm_test_extractor(
        res: Sequence[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pvals, corr_bs = zip(*res)
        pvals = np.sum(pvals, axis=0) / float(n_perms)

        corr_bs = np.concatenate(corr_bs, axis=0)
        corr_ci_low, corr_ci_high = np.quantile(corr_bs, q=ql, axis=0), np.quantile(
            corr_bs, q=qh, axis=0
        )

        return pvals, corr_ci_low, corr_ci_high

    if not (0 <= confidence_level <= 1):
        raise ValueError(
            f"Expected `confidence_level` to be in interval `[0, 1]`, found `{confidence_level}`."
        )

    n = X.shape[1]  # genes x cells
    ql = 1 - confidence_level - (1 - confidence_level) / 2.0
    qh = confidence_level + (1 - confidence_level) / 2.0

    if issparse(X) and not isspmatrix_csr(X):
        X = csr_matrix(X)

    corr = _mat_mat_corr_sparse(X, Y) if issparse(X) else _mat_mat_corr_dense(X, Y)

    # see: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Using_the_Fisher_transformation
    mean, se = np.arctanh(corr), 1.0 / np.sqrt(n - 3)
    z_score = (np.arctanh(corr) - np.arctanh(0)) * np.sqrt(n - 3)

    z = norm.ppf(qh)
    corr_ci_low = np.tanh(mean - z * se)
    corr_ci_high = np.tanh(mean + z * se)
    pvals = 2 * norm.cdf(-np.abs(z_score))

    return corr, pvals, corr_ci_low, corr_ci_high


def _mat_mat_corr_dense(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    n = X.shape[1]

    X_bar = np.reshape(np_mean(X, axis=1), (-1, 1))
    X_std = np.reshape(np_std(X, axis=1), (-1, 1))

    y_bar = np.reshape(np_mean(Y, axis=0), (1, -1))
    y_std = np.reshape(np_std(Y, axis=0), (1, -1))

    with np.warnings.catch_warnings():
        np.warnings.filterwarnings(
            "ignore", r"invalid value encountered in true_divide"
        )
        return (X @ Y - (n * X_bar * y_bar)) / ((n - 1) * X_std * y_std)


def _np_apply_along_axis(func1d, axis: int, arr: np.ndarray) -> np.ndarray:
    """
    Apply a reduction function over a given axis.
    Parameters
    ----------
    func1d
        Reduction function that operates only on 1 dimension.
    axis
        Axis over which to apply the reduction.
    arr
        The array to be reduced.
    Returns
    -------
    :class:`numpy.ndarray`
        The reduced array.
    """

    assert arr.ndim == 2
    assert axis in [0, 1]

    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
        return result

    result = np.empty(arr.shape[0])
    for i in range(len(result)):
        result[i] = func1d(arr[i, :])

    return result


def np_mean(array: np.ndarray, axis: int) -> np.ndarray:  # noqa
    return _np_apply_along_axis(np.mean, axis, array)


def np_std(array: np.ndarray, axis: int) -> np.ndarray:  # noqa
    return _np_apply_along_axis(np.std, axis, array)
