# """ Example code for running CCI analysis using new interface/approach.

# Tested: * Within-spot mode
#         * Between-spot mode

# TODO tests: * Above with cell heterogeneity information
# """

################################################################################
# Environment setup #
################################################################################
import stlearn as st
import matplotlib.pyplot as plt

################################################################################
# Load your data #
################################################################################
# TODO - load as an AnnData & perform usual pre-processing.
data = None  # replace with your code

# """ # Adding cell heterogeneity information if you have it.
# st.add.labels(data, 'tutorials/label_transfer_bc.csv', sep='\t')
# st.pl.cluster_plot(data, use_label="predictions")
# """

################################################################################
# Performing cci_rank analysis #
################################################################################
# Load the NATMI literature-curated database of LR pairs, data formatted #
lrs = st.tl.cci.load_lrs(["connectomeDB2020_lit"])

st.tl.cci.run(
    data,
    lrs,
    use_label=None,  # Need to add the label transfer results to object first, above code puts into 'label_transfer'
    use_het="cell_het",  # Slot for cell het. results in adata.obsm, only if use_label specified
    min_spots=6,  # Filter out any LR pairs with no scores for less than 6 spots
    distance=None,  # distance=0 for within-spot mode, None to auto-select distance to nearest neighbourhood.
    n_pairs=1000,  # Number of random pairs to generate
    adj_method="fdr_bh",  # MHT correction method
    min_expr=0,  # min expression for gene to be considered expressed.
    pval_adj_cutoff=0.05,
)
# """
# Example output:

# Calculating neighbours...
# 0 spots with no neighbours, 6 median spot neighbours.
# Spot neighbour indices stored in adata.uns['spot_neighbours']
# Altogether 1393 valid L-R pairs
# Generating random gene pairs...
# Generating the background...
# Calculating p-values for each LR pair in each spot...: 100%|██████████ [ time left: 00:00 ]

# Storing results:

# lr_scores stored in adata.obsm['lr_scores'].
# p_vals stored in adata.obsm['p_vals'].
# p_adjs stored in adata.obsm['p_adjs'].
# -log10(p_adjs) stored in adata.obsm['-log10(p_adjs)'].
# lr_sig_scores stored in adata.obsm['lr_sig_scores'].

# Per-spot results in adata.obsm have columns in same order as rows in adata.uns['lr_summary'].
# Summary of LR results in adata.uns['lr_summary'].
# """

################################################################################
# Visualising results #
################################################################################
# Plotting the -log10(p_adjs) for the lr with the highest number of spots.
# Set use_lr to any listed in data.uns['lr_summary'] to visualise alternate lrs.
st.pl.lr_result_plot(
    data,
    use_lr=None,  # Which LR to use, if None then uses top resuls from data.uns['lr_results']
    use_result="-log10(p_adjs)",  # Which result to visualise, must be one of
    # p_vals, p_adjs, -log10(p_adjs), lr_sig_scores
)
plt.show()

################################################################################
# Extra diagnostic plots for results #
################################################################################
# TODO:
#  Below needs to be updated with new way of storing results.

# Looking at which LR pairs were significant across the most spots #
print(data.uns["lr_summary"])  # Rank-ordered by pairs with most significant spots

# Now looking at the LR pair with the highest number of sig. spots #
best_lr = data.uns["lr_summary"].index.values[0]

# Binary LR coexpression plot for all spots #
st.pl.lr_plot(
    data,
    best_lr,
    inner_size_prop=0.1,
    outer_mode="binary",
    pt_scale=10,
    use_label=None,
    show_image=True,
    sig_spots=False,
)
plt.show()

# Significance scores for all spots #
st.pl.lr_plot(
    data,
    best_lr,
    inner_size_prop=1,
    outer_mode=None,
    pt_scale=20,
    use_label="lr_scores",
    show_image=True,
    sig_spots=False,
)
plt.show()

# Binary LR coexpression plot for significant spots #
st.pl.lr_plot(
    data,
    best_lr,
    outter_size_prop=1,
    outer_mode="binary",
    pt_scale=20,
    use_label=None,
    show_image=True,
    sig_spots=True,
)
plt.show()

# Continuous LR coexpression for signficant spots #
st.pl.lr_plot(
    data,
    best_lr,
    inner_size_prop=0.1,
    middle_size_prop=0.2,
    outter_size_prop=0.4,
    outer_mode="continuous",
    pt_scale=150,
    use_label=None,
    show_image=True,
    sig_spots=True,
)
plt.show()

# Continous LR coexpression for significant spots with tissue_type information #
st.pl.lr_plot(
    data,
    best_lr,
    inner_size_prop=0.08,
    middle_size_prop=0.3,
    outter_size_prop=0.5,
    outer_mode="continuous",
    pt_scale=150,
    use_label="tissue_type",
    show_image=True,
    sig_spots=True,
)
plt.show()


# # Old version of visualisation #
# """
# # LR enrichment scores
# data.obsm[f'{best_lr}_scores'] = data.uns['per_lr_results'][best_lr].loc[:,
#                                                              'lr_scores'].values
# # -log10(p_adj) of LR enrichment scores
# data.obsm[f'{best_lr}_log-p_adj'] = data.uns['per_lr_results'][best_lr].loc[:,
#                                                          '-log10(p_adj)'].values
# # Significant LR enrichment scores
# data.obsm[f'{best_lr}_sig-scores'] = data.uns['per_lr_results'][best_lr].loc[:,
#                                                          'lr_sig_scores'].values

# # Visualising these results #
# st.pl.het_plot(data, use_het=f'{best_lr}_scores', cell_alpha=0.7)
# plt.show()

# st.pl.het_plot(data, use_het=f'{best_lr}_sig-scores', cell_alpha=0.7)
# plt.show()
# """
