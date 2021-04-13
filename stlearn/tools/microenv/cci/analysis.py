""" Wrapper function for performing CCI analysis, varrying the analysis based on
    the inputted data / state of the anndata object.
"""

from .base import lr, calc_neighbours
from .het import count
from .merge import merge
from .permutation import permutation

def run(adata, lrs, use_label=None, distance=0, n_pairs=0, verbose=True,
        neg_binom: bool = False, adj_method: str = 'fdr', run_fast=False,
        bg_pairs = None,
        **kwargs):
    """Wrapper function for performing CCI analysis, varrying the analysis based 
        on the inputted data / state of the anndata object.
    Parameters
    ----------
    adata: AnnData          The data object including the cell types to count
    use_label:         The cell type results to use in counting
    use_het:                The stoarge place for result
    distance: int           Distance to determine the neighbours (default is the nearest neighbour), distance=0 means within spot
    **kwargs:               Extra arguments parsed to permutation.
    Returns
    -------
    adata: AnnData          With the counts of specified clusters in nearby spots stored as adata.uns['het']
    """
    adata.uns['lr'] = lrs
    neighbours = calc_neighbours(adata, distance,
                         index=True if 'fast' not in kwargs else kwargs['fast'])
    lr(adata=adata, distance=distance, neighbours=neighbours, **kwargs)

    # Conduct with cell heterogeneity info if label_transfer provided #
    cell_het = type(use_label)!=type(None) and use_label in adata.uns.keys()
    if cell_het:
        if verbose:
            print("Calculating cell hetereogeneity & merging with LR scores...")

        # Calculating cell heterogeneity #
        count(adata, distance=distance, use_label=use_label)

        # Merging with the lR values #
        merge(adata, use_lr='cci_lr', use_het='cci_het')

    if n_pairs != 0:  # Permutation testing #
        print("Performing permutation testing...")
        res = permutation(adata, use_het='cci_het' if cell_het else None,
                                   n_pairs=n_pairs, distance=distance,
                                   neg_binom=neg_binom, adj_method=adj_method,
                    neighbours=neighbours, run_fast=run_fast, bg_pairs=bg_pairs,
                          **kwargs)
        return res


