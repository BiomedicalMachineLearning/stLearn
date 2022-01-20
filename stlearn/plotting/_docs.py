doc_spatial_base_plot = """\
adata
    Annotated data matrix.
title
    Title name of the figure.
figsize
    Figure size with the format (width,height).
cmap
    Color map to use for continous variables or discretes variables (e.g. viridis, Set1,...).
use_label
    Key for the label use in `adata.obs` (e.g. `leiden`, `louvain`,...).
list_clusters
    A set of cluster to be displayed in the figure (e.g. [0,1,2,3]).
ax
    A matplotlib axes object.
show_plot
    Option to display the figure.
show_image
    Option to display the H&E image.
show_color_bar
    Option to display color bar.
crop
    Option to crop the figure based on the spot locations.
margin
    Margin to crop.
size
    Spot size to display in figure.
image_alpha
    Opacity of H&E image.
cell_alpha
    Opacity of spots/cells.
use_raw
    Option to use `adata.raw` data.
fname
    Output path to the output if user want to save the figure.
dpi
    Dots per inch values for the output.
"""

doc_gene_plot = """\
gene_symbols
    Single gene (str) or multiple genes (list) that user wants to display. It should be available in `adata.var_names`.
threshold
    Threshold to display genes in the figure.
method
    Method to combine multiple genes:
    `'CumSum'` is cummulative sum of genes expression values,
    `'NaiveMean'` is the mean of the genes expression values.
contour
    Option to show the contour plot.
step_size
    Determines the number and positions of the contour lines / regions.
"""

doc_cluster_plot = """\
show_subcluster
    Display the subcluster in the figure.
show_cluster_labels
    Display the labels of clusters.
show_trajectories
    Display the spatial trajectory analysis results.
reverse
    Reverse the direction of spatial trajectories.
show_node
    Show node of PAGA graph mapping in spatial.
threshold_spots
    The number of spots threshold for not display the subcluster labels
text_box_size
    The font size in the box of labels.
color_bar_size
    The size of color bar.
bbox_to_anchor
    Set the position of box of color bar. Default is `(1,1)`
"""

doc_lr_plot = """\
adata
    AnnData object with run st.tl.cci_rank.run performed on.
lr
    Ligand receptor paid (in format L_R)
min_expr
    Minimum expression for gene to be considered expressed.
sig_spots
    Whether to filter to significant spots or not.
use_label
    Label to use for the inner points, can be in adata.obs or in the lr stats of adata.uns['per_lr_results'][lr].columns
use_mix
    The deconvolution/label_transfer results to use for visualising pie charts in the inner point, not currently implimented.
outer_mode
    Either 'binary', 'continuous', or None; controls how ligand-receptor expression shown (or not shown).
l_cmap
    matplotlib cmap controlling ligand continous expression.
r_cmap
    matplotlib cmap controlling receptor continuous expression.
lr_cmap
    matplotlib cmap controlling the ligand receptor binary expression, but have atleast 4 colours.
inner_cmap
    matplotlib cmap controlling the inner point colours.
inner_size_prop
    multiplier which controls size of inner points.
middle_size_prop
    Multiplier which controls size of middle point (only relevant when outer_mode='continuous')
outer_size_prop
    Multiplier which controls size of the outter point.
pt_scale
    Multiplier which scales overall point size of all points plotted.
title
    Title of the plot.
show_image
    Whether to show the background H&E or not.
kwargs
    Extra arguments parsed to the other plotting functions such as gene_plot, cluster_plot, &/or het_plot.
"""

doc_het_plot = """\
use_het
    Single gene (str) or multiple genes (list) that user wants to display. It should be available in `adata.var_names`.
contour
    Option to show the contour plot.
step_size
    Determines the number and positions of the contour lines / regions.
vmin
    Lower end of scale bar.
vmax
    Upper end of scale bar.
"""

doc_subcluster_plot = """\
cluster
    Choose cluster to plot the sub-clusters.
text_box_size
    The font size in the box of labels.
bbox_to_anchor
    Set the position of box of color bar. Default is `(1,1)`
"""
