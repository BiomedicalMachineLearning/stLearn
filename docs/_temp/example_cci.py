""" Example code for running CCI analysis using new interface/approach.

Tested: * Within-spot mode
        * Between-spot mode

TODO tests: * Above with cell heterogeneity information
"""

################################################################################
                    # Environment setup #
################################################################################
import stlearn as st
import matplotlib.pyplot as plt

################################################################################
                    # Load your data #
################################################################################
# TODO - load as an AnnData & perform usual pre-processing.
data = None #replace with your code

""" # Adding cell heterogeneity information if you have it. 
st.add.labels(data, 'tutorials/label_transfer_bc.csv', sep='\t')
st.pl.cluster_plot(data, use_label="predictions")
"""

################################################################################
                # Performing cci analysis #
################################################################################
# Load the NATMI literature-curated database of LR pairs, human formatted #
lrs = st.tl.cci.load_lrs(['connectomeDB2020_lit'])

st.tl.cci.run(data, lrs,
              use_label = None, #Need to add the label transfer results to object first, above code puts into 'label_transfer'
              use_het = 'cell_het', #Slot for cell het. results in adata.obsm, only if use_label
              min_spots = 5, #Filter out any LR pairs with no scores for less than 5 spots
              distance=40, #distance=0 for within-spot mode
              n_pairs=200, #Number of random pairs to generate
              adj_method='fdr_bh', #MHT correction method
              lr_mid_dist = 200 #Controls how LR pairs grouped when creating bg distribs, higher number results in less groups
                                #Recommended to re-run a few times with different values to ensure results robust to this parameter.
              )
"""
Example output:

Calculating neighbours...
3 spots with no neighbours, 6 median spot neighbours.
Altogether 334 valid L-R pairs
18 lr groups with similar expression levels.
Generating background for each group, may take a while...
334 LR pairs with significant interactions.
Summary of significant spots for each lr pair in adata.uns['lr_summary'].
Spot enrichment statistics of LR interactions in adata.uns['per_lr_results']
"""

################################################################################
                    # Visualising results #
################################################################################
# Looking at which LR pairs were significant across the most spots #
print(data.uns['lr_summary']) #Rank-ordered by pairs with most significant spots

# Visualising the cell heterogeneity, if we included this information #
if 'cell_het' in data.obsm:
    st.pl.het_plot(data, use_het='merged', cell_alpha=0.7)
    plt.show()

# Now looking at the LR pair with the highest number of sig. spots #
best_lr = data.uns['lr_summary'].index.values[4]
# LR enrichment scores
data.obsm[f'{best_lr}_scores'] = data.uns['per_lr_results'][best_lr].loc[:,
                                                             'lr_scores'].values
# -log10(p_adj) of LR enrichment scores
data.obsm[f'{best_lr}_log-p_adj'] = data.uns['per_lr_results'][best_lr].loc[:,
                                                         '-log10(p_adj)'].values
# Significant LR enrichment scores
data.obsm[f'{best_lr}_sig-scores'] = data.uns['per_lr_results'][best_lr].loc[:,
                                                         'lr_sig_scores'].values

# Visualising these results #
st.pl.het_plot(data, use_het=f'{best_lr}_scores', cell_alpha=0.7)
plt.show()

st.pl.het_plot(data, use_het=f'{best_lr}_sig-scores', cell_alpha=0.7)
plt.show()

