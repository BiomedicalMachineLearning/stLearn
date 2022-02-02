"""
Functions for annotating spots as cell types.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc

import stlearn.tools.microenv.cci.r_helpers as rhs


def run_label_transfer(
    st_data, sc_data, sc_label_col, r_path, st_label_col=None, n_highly_variable=2000
):
    """Runs Seurat label transfer."""
    st_label_col = sc_label_col if type(st_label_col) == type(None) else st_label_col

    # Setting up the R environment #
    rhs.rpy2_setup(r_path)

    # Adding the source R code #
    r = rhs.ro.r
    path = os.path.dirname(os.path.realpath(__file__))
    r["source"](path + "/label_transfer.R")

    # Loading the label_transfer function #
    label_transfer_r = rhs.ro.globalenv["label_transfer"]
    print("Finished sourcing R code.")

    # Getting common gene set #
    sc_genes = sc_data.var_names
    st_genes = st_data.var_names
    genes = [gene for gene in sc_genes if gene in st_genes]
    sc_data = sc_data[:, genes].copy()
    st_data = st_data[:, genes].copy()

    # Extracting top & subsetting to top features to increase run-time speed #
    sc.pp.highly_variable_genes(
        st_data, n_top_genes=n_highly_variable, flavor="seurat_v3"
    )
    sc.pp.highly_variable_genes(
        sc_data, n_top_genes=n_highly_variable, flavor="seurat_v3"
    )
    genes_bool = np.logical_or(
        sc_data.var["highly_variable"].values, st_data.var["highly_variable"].values
    )
    sc_data = sc_data[:, genes_bool]
    st_data = st_data[:, genes_bool]
    print(
        f"Finished selecting & subsetting to hvgs: sc shape: "
        f"{sc_data.shape}, st shape: {st_data.shape}"
    )

    # Extracting the relevant information from anndatas #
    st_expr_df = st_data.to_df().transpose()
    sc_expr_df = sc_data.to_df().transpose()
    sc_labels = sc_data.obs[sc_label_col].values.astype(str)
    print(f"Finished extracting data. st shape: {st_expr_df.shape}")

    # R conversion of the data #
    sc_labels_r = rhs.ro.StrVector(sc_labels)
    with rhs.localconverter(rhs.ro.default_converter + rhs.pandas2ri.converter):
        # st_expr_df_r = rhs.ro.conversion.rpy2py(go_results_r)
        st_expr_df_r = rhs.ro.conversion.py2rpy(st_expr_df)
        sc_expr_df_r = rhs.ro.conversion.py2rpy(sc_expr_df)
    print("Finished py->rpy conversion.")

    # Running label transfer #
    label_transfer_scores_r = label_transfer_r(st_expr_df_r, sc_expr_df_r, sc_labels_r)
    print("Finished label transfer.")
    with rhs.localconverter(rhs.ro.default_converter + rhs.pandas2ri.converter):
        label_transfer_scores = rhs.ro.conversion.rpy2py(label_transfer_scores_r)
    print("Finished results rpy->py conversion.")

    # Adding to spatial anndata object #
    label_transfer_scores = label_transfer_scores.transpose()
    argmaxs = np.argmax(label_transfer_scores.values, axis=1)
    labels = [label_transfer_scores.columns.values[argmax] for argmax in argmaxs]

    st_data.obs[st_label_col] = labels
    st_data.obs[st_label_col] = st_data.obs[st_label_col].astype("category")
    label_transfer_scores.index = st_data.obs_names.values
    st_data.uns[st_label_col] = label_transfer_scores

    print(f"Spot labels added to st_data.obs[{st_label_col}].")
    print(f"Spot label scores added to st_data.uns[{st_label_col}].")


def get_counts(data):
    """Gets count data from anndata if available."""
    # Standard layer has counts #
    if type(data.X) != np.ndarray and np.all(np.mod(data.X[0, :].todense(), 1) == 0):
        counts = data.to_df().transpose()
    elif type(data.X) == np.ndarray and np.all(np.mod(data.X[0, :], 1) == 0):
        counts = data.to_df().transpose()
    elif (
        type(data.X) != np.ndarray
        and hasattr(data, "raw")
        and np.all(np.mod(data.raw.X[0, :].todense(), 1) == 0)
    ):
        counts = data.raw.to_adata()[data.obs_names, data.var_names].to_df().transpose()
    elif (
        type(data.X) == np.ndarray
        and hasattr(data, "raw")
        and np.all(np.mod(data.raw.X[0, :], 1) == 0)
    ):
        counts = data.raw.to_adata()[data.obs_names, data.var_names].to_df().transpose()
    else:
        raise Exception(
            "Inputted AnnData dosn't contain counts, only normalised "
            "values. Recreate with counts in .X or .raw.X."
        )

    return counts


def run_rctd(
    st_data,
    sc_data,
    sc_label_col,
    r_path,
    st_label_col=None,
    n_highly_variable=5000,
    min_cells=10,
    doublet_mode="full",
    n_cores=1,
):
    """Runs RCTD for deconvolution."""
    st_label_col = sc_label_col if type(st_label_col) == type(None) else st_label_col

    ########### Setting up the R environment #############
    rhs.rpy2_setup(r_path)

    # Adding the source R code #
    r = rhs.ro.r
    path = os.path.dirname(os.path.realpath(__file__))
    r["source"](path + "/rctd.R")

    # Loading the label_transfer function #
    rctd_r = rhs.ro.globalenv["rctd"]
    print("Finished sourcing R code.")

    # Getting common gene set #
    sc_genes = sc_data.var_names
    st_genes = st_data.var_names
    genes = [gene for gene in sc_genes if gene in st_genes]
    st_data_orig = st_data
    sc_data = sc_data[:, genes].copy()
    st_data = st_data[:, genes].copy()

    # Extracting top & subsetting to top features to increase run-time speed #
    sc.pp.highly_variable_genes(
        st_data, n_top_genes=n_highly_variable, flavor="seurat_v3"
    )
    sc.pp.highly_variable_genes(
        sc_data, n_top_genes=n_highly_variable, flavor="seurat_v3"
    )
    genes_bool = np.logical_or(
        sc_data.var["highly_variable"].values, st_data.var["highly_variable"].values
    )

    ###### Getting the count data (if available) ############
    st_counts = get_counts(st_data)
    sc_counts = get_counts(sc_data)

    st_counts = st_counts.loc[genes_bool, :]
    sc_counts = sc_counts.loc[genes_bool, :]

    st_coords = st_data.obs.loc[:, ["imagecol", "imagerow"]]
    sc_labels = sc_data.obs[sc_label_col].values.astype(str)
    print(f"Finished extracting counts data.")

    ####### Converting to R objects #########
    sc_labels_r = rhs.ro.StrVector(sc_labels)
    with rhs.localconverter(rhs.ro.default_converter + rhs.pandas2ri.converter):
        st_coords_r = rhs.ro.conversion.py2rpy(st_coords)
        st_counts_r = rhs.ro.conversion.py2rpy(st_counts)
        sc_counts_r = rhs.ro.conversion.py2rpy(sc_counts)
    print("Finished py->rpy conversion.")

    ######## Running RCTD ##########
    print("Running RCTD...")
    rctd_proportions_r = rctd_r(
        st_counts_r,
        st_coords_r,
        sc_counts_r,
        sc_labels_r,
        doublet_mode,
        min_cells,
        n_cores,
    )
    print("Finished RCTD deconvolution.")
    with rhs.localconverter(rhs.ro.default_converter + rhs.pandas2ri.converter):
        rctd_proportions = rhs.ro.conversion.rpy2py(rctd_proportions_r)
    print("Finished results rpy->py conversion.")

    # Adding to spatial anndata object #
    argmaxs = np.argmax(rctd_proportions.values, axis=1)
    labels = [rctd_proportions.columns.values[argmax] for argmax in argmaxs]
    st_data_orig.obs[st_label_col] = labels
    st_data_orig.obs[st_label_col] = st_data_orig.obs[st_label_col].astype("category")
    st_data_orig.uns[st_label_col] = rctd_proportions.loc[
        st_data_orig.obs_names.values, :
    ]

    print(f"Spot labels added to st_data.obs[{st_label_col}].")
    print(f"Spot label scores added to st_data.uns[{st_label_col}].")


def run_singleR(
    st_data,
    sc_data,
    sc_label_col,
    r_path,
    st_label_col=None,
    n_highly_variable=5000,
    n_centers=3,
    de_n=200,
    de_method="t",
):
    """Runs SingleR spot annotation."""
    st_label_col = sc_label_col if type(st_label_col) == type(None) else st_label_col
    ########### Setting up the R environment #############
    rhs.rpy2_setup(r_path)

    # Adding the source R code #
    r = rhs.ro.r
    path = os.path.dirname(os.path.realpath(__file__))
    r["source"](path + "/singleR.R")

    # Loading the label_transfer function #
    singleR_r = rhs.ro.globalenv["singleR"]
    print("Finished sourcing R code.")

    # Getting common gene set #
    sc_genes = sc_data.var_names
    st_genes = st_data.var_names
    genes = [gene for gene in sc_genes if gene in st_genes]
    st_data_orig = st_data
    sc_data = sc_data[:, genes].copy()
    st_data = st_data[:, genes].copy()

    # Extracting top & subsetting to top features to increase run-time speed #
    sc.pp.highly_variable_genes(
        st_data, n_top_genes=n_highly_variable, flavor="seurat_v3"
    )
    sc.pp.highly_variable_genes(
        sc_data, n_top_genes=n_highly_variable, flavor="seurat_v3"
    )
    genes_bool = np.logical_or(
        sc_data.var["highly_variable"].values, st_data.var["highly_variable"].values
    )
    sc_data = sc_data[:, genes_bool]
    st_data = st_data[:, genes_bool]
    print(f"Finished selecting & subsetting to hvgs.")

    # Extracting the relevant information from anndatas #
    st_expr_df = st_data.to_df().transpose()
    sc_expr_df = sc_data.to_df().transpose()
    sc_labels = sc_data.obs[sc_label_col].values.astype(str)
    print(f"Finished extracting data.")

    # R conversion of the data #
    sc_labels_r = rhs.ro.StrVector(sc_labels)
    de_method_r = rhs.ro.StrVector(de_method)
    with rhs.localconverter(rhs.ro.default_converter + rhs.pandas2ri.converter):
        st_expr_df_r = rhs.ro.conversion.py2rpy(st_expr_df)
        sc_expr_df_r = rhs.ro.conversion.py2rpy(sc_expr_df)
    print("Finished py->rpy conversion.")

    # Running label transfer #
    singleR_scores_r = singleR_r(
        st_expr_df_r, sc_expr_df_r, sc_labels_r, n_centers, de_n, de_method_r
    )
    print("Finished SingleR annotation.")
    with rhs.localconverter(rhs.ro.default_converter + rhs.pandas2ri.converter):
        singleR_scores = rhs.ro.conversion.rpy2py(singleR_scores_r)
    print("Finished results rpy->py conversion.")

    # Adding the results to the object #
    singleR_scores.index = [index.replace(".", "-") for index in singleR_scores.index]
    st_data_orig.obs[st_label_col] = singleR_scores.loc[:, "labels"].astype("category")
    singleR_scores_only = singleR_scores.drop(columns=["labels"])
    st_data_orig.uns[st_label_col] = singleR_scores_only

    print(f"Spot labels added to st_data.obs[{st_label_col}].")
    print(f"Spot label scores added to st_data.uns[{st_label_col}].")
