"""
Functions for annotating spots as cell types.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc

import stlearn.tools.microenv.cci.r_helpers as rhs

def run_label_transfer(st_data, sc_data, sc_label_col, r_path,
                       n_highly_variable=2000):
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
    print("Finished sourcing R code.")

    # Getting common gene set #
    sc_genes = sc_data.var_names
    st_genes = st_data.var_names
    genes = [gene for gene in sc_genes if gene in st_genes]
    sc_data = sc_data[:, genes].copy()
    st_data = st_data[:, genes].copy()

    # Extracting top & subsetting to top features to increase run-time speed #
    sc.pp.highly_variable_genes(st_data, n_top_genes=n_highly_variable,
                                flavor='seurat_v3')
    sc.pp.highly_variable_genes(sc_data, n_top_genes=n_highly_variable,
                                flavor='seurat_v3')
    sc_data = sc_data[:,sc_data.var['highly_variable']]
    st_data = st_data[:,st_data.var['highly_variable']]
    print(f"Finished selecting & subsetting to hvgs: sc shape: "
          f"{sc_data.shape}, st shape: {st_data.shape}")

    # Extracting the relevant information from anndatas #
    st_expr_df = st_data.to_df().transpose()
    sc_expr_df = sc_data.to_df().transpose()
    sc_labels = sc_data.obs[sc_label_col].values.astype(str)
    print(f"Finished extracting data. st shape: {st_expr_df.shape}")

    # R conversion of the data #
    sc_labels_r = rhs.ro.StrVector( sc_labels )
    with rhs.localconverter(rhs.ro.default_converter + rhs.pandas2ri.converter):
        #st_expr_df_r = rhs.ro.conversion.rpy2py(go_results_r)
        st_expr_df_r = rhs.ro.conversion.py2rpy(st_expr_df)
        sc_expr_df_r = rhs.ro.conversion.py2rpy(sc_expr_df)
    print('Finished py->rpy conversion.')
    print(st_expr_df_r)

    # Running label transfer #
    label_transfer_scores_r = label_transfer_r(st_expr_df_r, sc_expr_df_r,
                                                                    sc_labels_r)
    print("Finished label transfer.")
    with rhs.localconverter(rhs.ro.default_converter + rhs.pandas2ri.converter):
        label_transfer_scores = rhs.ro.conversion.rpy2py(label_transfer_scores_r)
    print("Finished results rpy->py conversion.")

    return label_transfer_scores








