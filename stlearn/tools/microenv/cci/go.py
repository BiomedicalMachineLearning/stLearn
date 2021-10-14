""" Wrapper for performing the LR GO analysis.
"""

import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ro = None
pandas2ri = None
localconverter = None

def rpy2_setup(r_path):
    """ Sets up rpy2.
    """
    os.environ['R_HOME'] = r_path

    import rpy2.robjects as robjects_
    from rpy2.robjects import pandas2ri as pandas2ri_
    from rpy2.robjects.conversion import localconverter as localconverter_
    global ro, pandas2ri, localconverter
    ro = robjects_
    pandas2ri = pandas2ri_
    localconverter = localconverter_

def run_GO(genes, species, r_path):
    """ Running GO term analysis.
    """
    # if 'R_HOME' not in os.environ:
    #     raise Exception("Need to run rpy2_setup(r_path) first !!")
    # Setting up the R environment #
    rpy2_setup(r_path)

    # Adding the source R code #
    r = ro.r
    path = os.path.dirname(os.path.realpath(__file__))
    r['source'](path+'/go.R')

    # Loading the GO analysis function #
    go_analyse_r = ro.globalenv['GO_analyse']

    # Running the function on the genes #
    genes_r = ro.StrVector(genes)
    go_results_r = go_analyse_r(genes_r, species)
    with localconverter(ro.default_converter + pandas2ri.converter):
        go_results = ro.conversion.rpy2py(go_results_r)

    return go_results




