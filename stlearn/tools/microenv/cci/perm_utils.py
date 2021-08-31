import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, canberra
from sklearn.preprocessing import MinMaxScaler

from numba import njit, prange
from numba.typed import List

def nonzero_quantile(expr, q, interpolation):
    """ Calculating the non-zero quantiles. """
    nonzero_expr = expr[ expr>0 ]
    quants = np.quantile(nonzero_expr, q=q, interpolation=interpolation)
    if type(quants) != np.array and type(quants) != np.ndarray:
        quants = np.array( [quants] )
    return quants

def getzero_prop(expr):
    """ Calculating the proportion of zeros. """
    zero_bool = expr==0
    n_zeros = len( np.where(zero_bool)[0] )
    zero_prop = [n_zeros / len(expr)]
    return zero_prop

def get_lr_quants(lr_expr: pd.DataFrame,
                  l_indices: list, r_indices: list, quantiles: np.array,
                  method=''):
    """ Gets the quantiles per gene in the LR pair, & then concatenates.
    Returns
    -------
    lr_quants, l_quants, r_quants: np.ndarray   First is concatenation of two latter. Each row is a quantile value, each column is a LR pair.
    """

    quant_func = nonzero_quantile if method!='quantiles' else np.quantile

    # First getting the quantiles of gene expression #
    gene_quants = np.apply_along_axis(quant_func, 0, lr_expr.values,
                                           q=quantiles, interpolation='nearest')

    l_quants = gene_quants[:, l_indices]
    r_quants = gene_quants[:, r_indices]

    lr_quants = np.concatenate((l_quants, r_quants), 0).transpose()

    return lr_quants, l_quants, r_quants

def get_lr_zeroprops(lr_expr: pd.DataFrame, l_indices: list, r_indices: list):
    """ Gets the proportion of zeros per gene in the LR pair, & then concatenates.
    Returns
    -------
    lr_props, l_props, r_props: np.ndarray   First is concatenation of two latter. Each row is a prop value, each column is a LR pair.
    """

    # First getting the quantiles of gene expression #
    gene_props = np.apply_along_axis(getzero_prop, 0, lr_expr.values)

    l_props = gene_props[:, l_indices]
    r_props = gene_props[:, r_indices]

    lr_props = np.concatenate((l_props, r_props), 0).transpose()

    return lr_props, l_props, r_props

def get_lr_bounds(lr_value: float, bin_bounds: np.array):
    """ For the given lr_value, returns the bin where it belongs.
    Returns
    -------
    lr_bin: tuple   Tuple of length 2, first is the lower bound of the bin, second is upper bound of the bin.
    """
    if np.any(bin_bounds == lr_value): # If sits on a boundary
        lr_i = np.where(bin_bounds == lr_value)[0][0]
        if lr_value == max(bin_bounds): # Must be in the final bin
            _lower = bin_bounds[-2]
            _upper = bin_bounds[-1]
        else: # In the lower bin
            _lower = bin_bounds[lr_i]
            _upper = bin_bounds[lr_i + 1]
    else: # Bin where it's value sit in-between #
        _lower = bin_bounds[np.where(bin_bounds < lr_value)[0][-1]]
        _upper = bin_bounds[np.where(bin_bounds > lr_value)[0][0]]

    return (_lower, _upper)

def get_similar_genes(ref_quants: np.array, ref_props: np.array, n_genes: int,
                      candidate_expr: np.ndarray, candidate_genes: np.array,
                      quantiles=(.5),#(.5, .75, .85, .9, .95, .97, .98, .99, 1)
                      ):
    """ Gets genes with a similar expression distribution as the inputted gene,
        by measuring distance between the gene expression quantiles.
    Parameters
    ----------
    ref_quants: np.array     The pre-calculated quantiles.
    ref_props: np.array      The query zero proportions.
    n_genes: int            Number of equivalent genes to select.
    candidate_expr: np.ndarray  Expression of gene candidates (cells*genes).
    candidate_genes: np.array   Same as candidate_expr.shape[1], indicating gene names.
    quantiles: tuple    The quantile to use
    Returns
    -------
    similar_genes: np.array Array of strings for gene names.
    """
    if type(quantiles)==float:
        quantiles = np.array([quantiles])
    else:
        quantiles = np.array(quantiles)

    # Query quants #
    query_quants = np.apply_along_axis(nonzero_quantile, 0, candidate_expr,
                                           q=quantiles, interpolation='nearest')

    # Need to min-max normalise so can take distance with the proportion #
    all_quants = np.concatenate((np.array([ref_quants]), query_quants), axis=1)
    scaler = MinMaxScaler()
    scaled_quants = scaler.fit_transform(all_quants.transpose()).transpose()
    ref_scaled = scaled_quants[:, 0]
    query_scaled = scaled_quants[:, 1:]

    # Query props #
    query_props = np.apply_along_axis(getzero_prop, 0, candidate_expr)

    # Concatenating to create the ref & query vals to match #
    ref_vals = np.array([ref_scaled[0], ref_props[0]]) # both between 0 & 1
    query_vals = np.concatenate((query_scaled, query_props))

    # Measuring distances from the desired gene #
    dists = np.apply_along_axis(canberra, 0, query_vals, ref_vals)

    # Retrieving desired number of genes #
    order = np.argsort(dists)
    similar_genes = candidate_genes[order[0:n_genes]]

    """ During debugging, plotting distribution of distances & selected genes.
    import matplotlib.pyplot as plt
    cutoff = dists[order[n_genes]]
    plt.hist(dists, bins=100)
    plt.vlines(cutoff, 0, 500, color='r')
    plt.show()
    """

    return similar_genes

def get_similar_genes_Quantiles(gene_expr: np.array, n_genes: int,
                      candidate_quants: np.ndarray, candidate_genes: np.array,
                      quantiles=(.5, .75, .85, .9, .95, .97, .98, .99, 1)):
    """ Gets genes with a similar expression distribution as the inputted gene,
        by measuring distance between the gene expression quantiles.
    Parameters
    ----------
    gene_expr: np.array     Expression of the gene of interest, or, if the same length as quantiles, then assumes is the pre-calculated quantiles.
    n_genes: int            Number of equivalent genes to select.
    candidate_quants: np.ndarray  Expression quantiles of gene candidates (quantiles*genes).
    candidate_genes: np.array   Same as candidate_expr.shape[1], indicating gene names.
    quantiles: tuple    The quantile to use
    Returns
    -------
    similar_genes: np.array Array of strings for gene names.
    """

    if type(quantiles)==float:
        quantiles = np.array([quantiles])
    else:
        quantiles = np.array(quantiles)

    # Getting the quantiles for the gene #
    if len(gene_expr) != len(quantiles):
        #ref_quants = np.quantile(gene_expr, q=quantiles, interpolation='nearest')
        ref_quants = nonzero_quantile(gene_expr, q=quantiles,
                                                        interpolation='nearest')
    else:
        ref_quants = gene_expr

    # Measuring distances from the desired gene #
    dists = np.apply_along_axis(canberra, 0, candidate_quants, ref_quants)
    order = np.argsort(dists)

    """ During debugging, plotting distribution of distances & selected genes.
    import matplotlib.pyplot as plt
    cutoff = dists[order[n_genes]]
    fig, ax = plt.subplots()
    ax.hist(dists[order[0:28000]], bins=1000)
    y_max = ax.get_ylim()[1]
    ax.vlines(cutoff, 0, y_max/2, color='r')
    plt.show()
    print(candidate_quants[:,order[0:3]]) # Showing the quantiles of selected
    print(candidate_quants[:,order[n_genes-3:n_genes]])
    print(ref_quants)
    """

    # Retrieving desired number of genes #
    similar_genes = candidate_genes[order[0:n_genes]]

    return similar_genes

@njit(parallel=True)
def get_similar_genesFAST(ref_quants: np.array, n_genes: int,
                       candidate_quants: np.ndarray, candidate_genes: np.array):
    """Fast version of the above with parallelisation."""

    # Measuring distances from the desired gene #
    dists = np.zeros((1, candidate_quants.shape[1]), dtype=np.float64)[0,:]
    for i in prange(0, candidate_quants.shape[1]):
        cand_quants = candidate_quants[:,i]
        abs_diff = np.abs(ref_quants - cand_quants)
        dists[i] = np.nansum( abs_diff / (ref_quants + cand_quants) )

    # Need to remove the zero-dists since this indicates they are expressed
    # exactly the same, & hence likely in the same spot !!!
    nonzero_bool = dists > 0
    dists = dists[nonzero_bool]
    candidate_quants = candidate_quants[:,nonzero_bool]
    candidate_genes = candidate_genes[nonzero_bool]
    order = np.argsort(dists)

    """ During debugging, plotting distribution of distances & selected genes.
    import matplotlib.pyplot as plt
    cutoff = dists[order[n_genes]]
    fig, ax = plt.subplots()
    ax.hist(dists[order[0:28000]], bins=1000)
    y_max = ax.get_ylim()[1]
    ax.vlines(cutoff, 0, y_max/2, color='r')
    plt.show()
    print(candidate_quants[:,order[0:3]]) # Showing the quantiles of selected
    print(candidate_quants[:,order[n_genes-3:n_genes]])
    print(ref_quants)
    """

    # Retrieving desired number of genes #
    similar_genes = candidate_genes[order[0:n_genes]]

    return similar_genes

@njit
def gen_rand_pairs(genes1: np.array, genes2: np.array, n_pairs: int):
    """Generates random pairs of genes."""
    # genes1_ins = np.array(list(range(genes1)))
    # genes2_ins = np.array(list(range(genes2)))

    rand_pairs = List() #np.zeros((n_pairs, 2), dtype=np.float)
    for j in prange(0, n_pairs):
        l_rand = np.random.choice(genes1, 1)[0]
        r_rand = np.random.choice(genes2, 1)[0]
        rand_pair = '_'.join([l_rand, r_rand])
        # l_rand_i = np.random.choice(genes1_ins, 1)
        # r_rand_i = np.random.choice(genes2_ins, 1)
        while rand_pair in rand_pairs and l_rand == r_rand:
            l_rand = np.random.choice(genes1, 1)[0]
            r_rand = np.random.choice(genes2, 1)[0]
            rand_pair = '_'.join([l_rand, r_rand])
        #rand_pairs[j] = rand_pair
        rand_pairs.append(rand_pair)

    return rand_pairs

