"""
Functions for annotating spots as cell types.
"""

import os
import numpy as np
import pandas as pd

import stlearn.tools.microenv.cci.r_helpers as rhs

def run_label_transfer(st_data, sc_data, sc_label_col, r_path):
    """ Runs Seurat label transfer.
    """
    # Setting up the R environment #
    rhs.rpy2_setup(r_path)

    # Adding the source R code #
    r = rhs.ro.r
    path = os.path.dirname(os.path.realpath(__file__))
    r['source'](path+'/label_transfer.R')

    # Loading the label_transfer function #
    label_transfer_r = rhs.ro.globalenv['label_transfer']

    # Extracting the relevant information from anndatas #
    st_expr_df = st_data.to_df().transpose()
    sc_expr_df = sc_data.to_df().transpose()
    sc_labels = sc_data.obs[sc_label_col].values.astype(str)

    # R conversion of the data #
    sc_labels_r = rhs.ro.StrVector( sc_labels )
    with rhs.localconverter(rhs.ro.default_converter + rhs.pandas2ri.converter):
        #st_expr_df_r = rhs.ro.conversion.rpy2py(go_results_r)
        st_expr_df_r = rhs.ro.conversion.py2rpy(st_expr_df)
        sc_expr_df_r = rhs.ro.conversion.py2rpy(sc_expr_df)

    # Running label transfer #
    label_transfer_scores_r = label_transfer_r(st_expr_df_r, sc_expr_df_r,
                                                                    sc_labels_r)
    with rhs.localconverter(rhs.ro.default_converter + rhs.pandas2ri.converter):
        label_transfer_scores = rhs.ro.conversion.rpy2py(label_transfer_scores_r)

    return label_transfer_scores








