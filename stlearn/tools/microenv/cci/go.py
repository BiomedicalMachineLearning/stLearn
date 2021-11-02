""" Wrapper for performing the LR GO analysis.
"""

import os
import stlearn.tools.microenv.cci.r_helpers as rhs

def run_GO(genes, bg_genes, species, r_path):
    """ Running GO term analysis.
    """

    # Setting up the R environment #
    rhs.rpy2_setup(r_path)

    # Adding the source R code #
    r = rhs.ro.r
    path = os.path.dirname(os.path.realpath(__file__))
    r['source'](path+'/go.R')

    # Loading the GO analysis function #
    go_analyse_r = rhs.ro.globalenv['GO_analyse']

    # Running the function on the genes #
    genes_r = rhs.ro.StrVector( genes )
    bg_genes_r = rhs.ro.StrVector( bg_genes )
    go_results_r = go_analyse_r(genes_r, bg_genes_r, species)
    with rhs.localconverter(rhs.ro.default_converter + rhs.pandas2ri.converter):
        go_results = rhs.ro.conversion.rpy2py(go_results_r)

    return go_results




