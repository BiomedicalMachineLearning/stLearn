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

doc_het_plot = """\
use_het
    Single gene (str) or multiple genes (list) that user wants to display. It should be available in `adata.var_names`.
contour
    Option to show the contour plot.
step_size
    Determines the number and positions of the contour lines / regions.
"""

doc_subcluster_plot = """\
cluster
    Choose cluster to plot the sub-clusters.
text_box_size
    The font size in the box of labels.
bbox_to_anchor
    Set the position of box of color bar. Default is `(1,1)`
"""
