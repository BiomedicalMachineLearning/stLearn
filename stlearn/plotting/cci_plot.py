from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib
import pandas as pd
import numpy as np
import networkx as nx
import math
import matplotlib.patches as patches
from numba.typed import List
import seaborn as sns
import sys
from anndata import AnnData
from typing import Optional, Union

from typing import Optional, Union, Mapping  # Special
from typing import Sequence, Iterable  # ABCs
from typing import Tuple  # Classes

import warnings

from .classes import CciPlot, LrResultPlot
from .classes_bokeh import BokehCciPlot
from ._docs import doc_spatial_base_plot, doc_het_plot, doc_lr_plot
from ..utils import Empty, _empty, _AxesSubplot, _docs_params
from .utils import get_cmap, check_cmap
from .cluster_plot import cluster_plot
from .deconvolution_plot import deconvolution_plot
from .gene_plot import gene_plot
from stlearn.plotting.utils import get_colors
import stlearn.plotting.cci_plot_helpers as cci_hs
from .cci_plot_helpers import get_int_df, add_arrows, create_flat_df, _box_map, \
																	chordDiagram

import importlib
importlib.reload(cci_hs)

from bokeh.io import push_notebook, output_notebook
from bokeh.plotting import show

""" Functions for visualisation the LR results per spot. 
"""

def lr_result_plot(
		adata: AnnData,
		use_lr: Optional["str"] = None,
		use_result: Optional["str"] = "lr_sig_scores",
		# plotting param
		title: Optional["str"] = None,
		figsize: Optional[Tuple[float, float]] = None,
		cmap: Optional[str] = "Spectral_r",
		list_clusters: Optional[list] = None,
		ax: Optional[matplotlib.axes._subplots.Axes] = None,
		fig: Optional[matplotlib.figure.Figure] = None,
		show_plot: Optional[bool] = True,
		show_axis: Optional[bool] = False,
		show_image: Optional[bool] = True,
		show_color_bar: Optional[bool] = True,
		crop: Optional[bool] = True,
		margin: Optional[bool] = 100,
		size: Optional[float] = 7,
		image_alpha: Optional[float] = 1.0,
		cell_alpha: Optional[float] = 1.0,
		use_raw: Optional[bool] = False,
		fname: Optional[str] = None,
		dpi: Optional[int] = 120,
		# cci param
		contour: bool = False,
		step_size: Optional[int] = None,
		vmin: float = None, vmax: float = None,
):
	LrResultPlot(
		adata,
		use_lr,
		use_result,
		# plotting param
		title,
		figsize,
		cmap,
		list_clusters,
		ax,
		fig,
		show_plot,
		show_axis,
		show_image,
		show_color_bar,
		crop,
		margin,
		size,
		image_alpha,
		cell_alpha,
		use_raw,
		fname,
		dpi,
		# cci param
		contour,
		step_size,
		vmin, vmax,
	)

#@_docs_params(het_plot=doc_lr_plot)
def lr_plot(
	adata: AnnData, lr: str,
	min_expr: float = 0, sig_spots=True,
	use_label: str = None, use_mix: str = None, outer_mode: str = 'continuous',
	l_cmap=None, r_cmap=None, lr_cmap=None, inner_cmap=None,
	inner_size_prop: float=0.25, middle_size_prop: float=0.5,
	outer_size_prop: float=1, pt_scale: int=100, title='',
	show_image: bool=True, show_arrows: bool=False,
	fig: Figure = None, ax: Axes=None,
	arrow_head_width: float=4, arrow_width: float=.001, arrow_cmap: str=None,
        arrow_vmax: float=None,
		sig_cci: bool=False, lr_colors: dict=None,
	# plotting params
	**kwargs,
) -> Optional[AnnData]:

	# Input checking #
	l, r = lr.split('_')
	ran_lr = 'lr_summary' in adata.uns
	ran_sig = False if not ran_lr else 'n_spots_sig' in adata.uns['lr_summary'].columns
	if ran_lr and lr in adata.uns['lr_summary'].index:
		if ran_sig:
			lr_sig = adata.uns['lr_summary'].loc[lr, :].values[1] > 0
		else:
			lr_sig = True
	else:
		lr_sig = False

	if sig_spots and not ran_lr:
		raise Exception("No LR results testing results found, "
					  "please run st.tl.cci.run first, or set sig_spots=False.")

	elif sig_spots and not lr_sig:
		raise Exception("LR has no significant spots, to visualise anyhow set"
						"sig_spots=False")

	# Making sure have run_cci first with respective labelling #
	if show_arrows and sig_cci and use_label and f'per_lr_cci_{use_label}' \
		not in adata.uns:
		raise Exception("Cannot subset arrow interactions to significant ccis "
						"without performing st.tl.run_cci with "
						f"use_label={use_label} first.")

	# Getting which are the allowed stats for the lr to plot #
	if not ran_sig:
		lr_use_labels = ['lr_scores']
	else:
		lr_use_labels = ['lr_scores', 'p_val', 'p_adj',
						 '-log10(p_adj)', 'lr_sig_scores']

	if type(use_mix)!=type(None) and use_mix not in adata.uns:
		raise Exception(f"Specified use_mix, but no deconvolution results added "
					   "to adata.uns matching the use_mix ({use_mix}) key.")
	elif type(use_label)!=type(None) and use_label in lr_use_labels \
			and ran_sig and not lr_sig:
		raise Exception(f"Since use_label refers to lr stats & ran permutation testing, "
						f"LR needs to be significant to view stats.")
	elif type(use_label)!=type(None) and use_label not in adata.obs.keys() \
											 and use_label not in lr_use_labels:
		raise Exception(f"use_label must be in adata.obs or "
						f"one of lr stats: {lr_use_labels}.")

	out_options = ['binary', 'continuous', None]
	if outer_mode not in out_options:
		raise Exception(f"{outer_mode} should be one of {out_options}")

	if l not in adata.var_names or r not in adata.var_names:
		raise Exception("L or R not found in adata.var_names.")

	# Whether to show just the significant spots or all spots
	lr_index = np.where(adata.uns['lr_summary'].index.values == lr)[0][0]
	sig_bool = adata.obsm['lr_sig_scores'][:, lr_index] > 0
	if sig_spots:
		adata_full = adata
		adata = adata[sig_bool,:]
	else:
		adata_full = adata

	# Dealing with the axis #
	if type(fig)==type(None) or type(ax)==type(None):
		fig, ax = plt.subplots()

	expr = adata.to_df()
	l_expr = expr.loc[:, l].values
	r_expr = expr.loc[:, r].values
	# Adding binary points of the ligand/receptor pair #
	if outer_mode == 'binary':
		l_bool, r_bool = l_expr > min_expr, r_expr > min_expr
		lr_binary_labels = []
		for i in range(len(l_bool)):
			if l_bool[i] and not r_bool[i]:
				lr_binary_labels.append( l )
			elif not l_bool[i] and r_bool[i]:
				lr_binary_labels.append( r )
			elif l_bool[i] and r_bool[i]:
				lr_binary_labels.append( lr )
			elif not l_bool[i] and not r_bool[i]:
				lr_binary_labels.append( '' )
		lr_binary_labels = pd.Series(np.array(lr_binary_labels),
									   index=adata.obs_names).astype('category')
		adata.obs[f'{lr}_binary_labels'] = lr_binary_labels

		if type(lr_cmap) == type(None):
			lr_cmap = "default" #This gets ignored due to setting colours below
			if type(lr_colors)==type(None):
				lr_colors = {l: matplotlib.colors.to_hex('r'),
						  r: matplotlib.colors.to_hex('limegreen'),
						  lr: matplotlib.colors.to_hex('b'),
						  '': '#836BC6' # Neutral color in H&E images.
							 }

			label_set = adata.obs[f'{lr}_binary_labels'].cat.categories
			adata.uns[f'{lr}_binary_labels_colors'] = [lr_colors[label]
														 for label in label_set]
		else:
			lr_cmap = check_cmap(lr_cmap)

		cluster_plot(adata, use_label=f'{lr}_binary_labels', cmap=lr_cmap,
						   size=outer_size_prop * pt_scale, crop=False,
						   ax=ax, fig=fig, show_image=show_image, **kwargs)

	# Showing continuous gene expression of the LR pair #
	elif outer_mode == 'continuous':
		if type(l_cmap)==type(None):
			l_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('lcmap',
																[(0, 0, 0),
																 (.5, 0, 0),
																 (.75, 0, 0),
																 (1, 0, 0)])
		else:
			l_cmap = check_cmap(l_cmap)
		if type(r_cmap)==type(None):
			r_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('rcmap',
																[(0, 0, 0),
																 (0, .5, 0),
																 (0, .75, 0),
																 (0, 1, 0)])
		else:
			r_cmap = check_cmap(r_cmap)

		gene_plot(adata, gene_symbols=l, size=outer_size_prop * pt_scale,
			   cmap=l_cmap, color_bar_label=l, ax=ax, fig=fig, crop=False,
												show_image=show_image, **kwargs)
		gene_plot(adata, gene_symbols=r, size=middle_size_prop * pt_scale,
			   cmap=r_cmap, color_bar_label=r, ax=ax, fig=fig, crop=False,
												show_image=show_image, **kwargs)

	# Adding the cell type labels #
	if type(use_label) != type(None):
		if use_label in lr_use_labels:
			inner_cmap = inner_cmap if type(inner_cmap) != type(None) else "copper"
			# adata.obsm[f'{lr}_{use_label}'] = adata.uns['per_lr_results'][
			#                          lr].loc[adata.obs_names,use_label].values
			lr_result_plot(adata, use_lr=lr, show_image=show_image,
					 cmap=inner_cmap, crop=False,
					 ax=ax, fig=fig, size=inner_size_prop * pt_scale, **kwargs)
		else:
			inner_cmap = inner_cmap if type(inner_cmap)!=type(None) else "default"
			cluster_plot(adata, use_label=use_label, cmap=inner_cmap,
						 size=inner_size_prop * pt_scale, crop=False,
						 ax=ax, fig=fig, show_image=show_image, **kwargs)

	# Adding in labels which show the interactions between signicant spots &
	# neighbours
	if show_arrows:
		l_expr = adata_full[:, l].X.toarray()[:, 0]
		r_expr = adata_full[:, r].X.toarray()[:, 0]

		if sig_cci:
			int_df = adata.uns[f'per_lr_cci_{use_label}'][lr]
		else:
			int_df = None

		cci_hs.add_arrows(adata_full, l_expr, r_expr,
                          min_expr, sig_bool, fig, ax, use_label, int_df,
                          arrow_head_width, arrow_width, arrow_cmap, arrow_vmax)

	# Cropping #
	# if crop:
	#     x0, x1 = ax.get_xlim()
	#     y0, y1 = ax.get_ylim()
	#     x_margin, y_margin = (x1-x0)*margin_ratio, (y1-y0)*margin_ratio
	#     print(x_margin, y_margin)
	#     print(x0, x1, y0, y1)
	#     ax.set_xlim(x0 - x_margin, x1 + x_margin)
	#     ax.set_ylim(y0 - y_margin, y1 + y_margin)
	#     #ax.set_ylim(ax.get_ylim()[::-1])

	fig.suptitle(title)

#### het_plot currently out of date;
#### from old data structure when only test individual LRs.
@_docs_params(spatial_base_plot=doc_spatial_base_plot, het_plot=doc_het_plot)
def het_plot(
	adata: AnnData,
	# plotting param
	title: Optional["str"] = None,
	figsize: Optional[Tuple[float, float]] = None,
	cmap: Optional[str] = "Spectral_r",
	use_label: Optional[str] = None,
	list_clusters: Optional[list] = None,
	ax: Optional[matplotlib.axes._subplots.Axes] = None,
	fig: Optional[matplotlib.figure.Figure] = None,
	show_plot: Optional[bool] = True,
	show_axis: Optional[bool] = False,
	show_image: Optional[bool] = True,
	show_color_bar: Optional[bool] = True,
	crop: Optional[bool] = True,
	margin: Optional[bool] = 100,
	size: Optional[float] = 7,
	image_alpha: Optional[float] = 1.0,
	cell_alpha: Optional[float] = 1.0,
	use_raw: Optional[bool] = False,
	fname: Optional[str] = None,
	dpi: Optional[int] = 120,
	# cci param
	use_het: Optional[str] = "het",
	contour: bool = False,
	step_size: Optional[int] = None,
	vmin: float = None, vmax: float = None,
) -> Optional[AnnData]:
	"""\
	Allows the visualization of significant cell-cell interaction
	as the values of dot points or contour in the Spatial
	transcriptomics array.


	Parameters
	-------------------------------------
	{spatial_base_plot}
	{het_plot}

	Examples
	-------------------------------------
	>>> import stlearn as st
	>>> adata = st.datasets.example_bcba()
	>>> pvalues = "lr_pvalues"
	>>> st.pl.gene_plot(adata, use_het = pvalues)

	"""

	CciPlot(
		adata,
		title=title,
		figsize=figsize,
		cmap=cmap,
		use_label=use_label,
		list_clusters=list_clusters,
		ax=ax,
		fig=fig,
		show_plot=show_plot,
		show_axis=show_axis,
		show_image=show_image,
		show_color_bar=show_color_bar,
		crop=crop,
		margin=margin,
		size=size,
		image_alpha=image_alpha,
		cell_alpha=cell_alpha,
		use_raw=use_raw,
		fname=fname,
		dpi=dpi,
		use_het=use_het,
		contour=contour,
		step_size=step_size,
		vmin=vmin, vmax=vmax,
	)

""" Functions relating to visualising celltype-celltype interactions after 
	calling: st.tl.cci.run_cci
"""

def ccinet_plot(adata: AnnData, use_label: str, lr: str = None,
				pos: dict = None, return_pos: bool = False, cmap: str='default',
				font_size: int=12, node_size_exp: int=1, node_size_scaler: int=1,
				min_counts: int=0, sig_interactions: bool=True,
				fig: matplotlib.figure.Figure=None,
				ax: matplotlib.axes.Axes=None, pad=.25,
				title: str=None, figsize: tuple=(10,10),):
	""" Circular celltype-celltype interaction network based on LR analysis.
		Parameters
		----------
		adata: AnnData
		use_label: str    Indicates the cell type labels or deconvolution results use for cell-cell interaction counting by LR pairs.
		lr: str    The LR pair to visualise the cci network for. If None, will use all pairs via adata.uns[f'lr_cci_{use_label}'].
		pos: dict   Positions to draw each cell type, format as outputted from running networkx.circular_layout(graph). If not inputted will be generated.
		return_pos: bool   Whether to return the positions of the cell types drawn or not.
		cmap: str    Cmap to use when generating the cell colors, if not already specified by adata.uns[f'{use_label}_colors'].
		font_size: int    Size of the cell type labels.
		node_size_scaler: float   Scaler to multiply by node sizes to increase/decrease size.
		node_size_exp: int    Increases difference between node sizes by this exponent.
		min_counts: int    Minimum no. of LR interactions for connection to be drawn.

		Returns
		-------
		pos: dict    Dictionary of positions where the nodes are draw if return_pos is True, useful for consistent layouts.
	"""
	cmap, cmap_n = get_cmap(cmap)
	# Making sure adata in correct state that this function should run #
	if f'lr_cci_{use_label}' not in adata.uns:
		raise Exception("Need to first call st.tl.run_cci with the equivalnt "
						"use_label to visualise cell-cell interactions.")
	elif type(lr) != type(None) and \
			  lr not in adata.uns[f'per_lr_cci_{use_label}']:
		raise Exception(f"{lr} not found in {f'per_lr_cci_{use_label}'}, "
						"suggesting no significant interactions.")

	# Either plotting overall interactions, or just for a particular LR #
	int_df, title = get_int_df(adata, lr, use_label, sig_interactions, title)

	# Creating the interaction graph #
	all_set = int_df.index.values
	int_matrix = int_df.values
	graph = nx.MultiDiGraph()
	int_bool = int_matrix > min_counts
	int_matrix = int_matrix * int_bool
	for i, cell_A in enumerate(all_set):
		if cell_A not in graph:
			graph.add_node(cell_A)
		for j, cell_B in enumerate(all_set):
			if int_bool[i, j]:
				count = int_matrix[i, j]
				graph.add_edge(cell_A, cell_B, weight=count)

	# Determining graph layout, node sizes, & edge colours #
	if type(pos) == type(None):
		pos = nx.circular_layout(graph)  # position the nodes using the layout
	total = sum(sum(int_matrix))
	node_names = list(graph.nodes.keys())
	node_indices = [np.where(all_set==node_name)[0][0]
													for node_name in node_names]
	node_sizes = np.array([(((sum(int_matrix[i,:]+int_matrix[:,i])-
				int_matrix[i,i])/total)*10000*node_size_scaler)**(node_size_exp)
														 for i in node_indices])
	node_sizes[node_sizes==0] = .1 #pseudocount

	edges = list(graph.edges.items())
	e_totals = []
	for i, edge in enumerate(edges):
		trans_i = np.where(all_set==edge[0][0])[0][0]
		receive_i = np.where(all_set==edge[0][1])[0][0]
		e_total = sum(list(int_matrix[trans_i,:])+
					  list(int_matrix[:,receive_i]))\
				  -int_matrix[trans_i,receive_i] #so don't double count
		e_totals.append( e_total )
	edge_weights = [edge[1]['weight']/e_totals[i]
					for i, edge in enumerate(edges)]

	# Determining node colors #
	nodes = np.unique(list(graph.nodes.keys()))
	node_colors = get_colors(adata, 'cell_type', cmap, label_set=nodes)
	if not np.all(np.array(node_names)==nodes):
		nodes_indices = [np.where(nodes == node)[0][0]
						 for node in node_names]
		node_colors = np.array(node_colors)[nodes_indices]

	#### Drawing the graph #####
	if type(fig)==type(None) or type(ax)==type(None):
		fig, ax = plt.subplots(figsize=figsize, facecolor=[0.7, 0.7, 0.7, 0.4])

	# Adding in the self-loops #
	z = 55
	for i, edge in enumerate(edges):
		cell_type = edge[0][0]
		if cell_type != edge[0][1]:
			continue
		x, y = pos[cell_type]
		angle = math.degrees(math.atan(y / x))
		if x > 0:
			angle = angle+180
		arc = patches.Arc(xy=(x, y),
						  width=.3, height=.025, lw=5,
						  ec=plt.cm.get_cmap('Blues')(edge_weights[i]),
						  angle=angle, theta1=z, theta2=360 - z
						  )
		ax.add_patch(arc)

	# Drawing the main components of the graph #
	edges = nx.draw_networkx(
		graph,
		pos,
		node_size=node_sizes,
		node_color=node_colors,
		arrowstyle="->",
		arrowsize=50,
		width=5,
		font_size=font_size,
		font_weight='bold',
		edge_color=edge_weights,
		edge_cmap=plt.cm.Blues,
		ax=ax,
	)
	fig.suptitle(title, fontsize=30)
	plt.tight_layout()

	# Adding padding #
	xlims = ax.get_xlim()
	ax.set_xlim(xlims[0]-pad, xlims[1]+pad)
	ylims = ax.get_ylim()
	ax.set_ylim(ylims[0]-pad, ylims[1]+pad)

	if return_pos:
		return pos

def cci_map(adata: AnnData, use_label: str, lr: str=None,
			ax: matplotlib.figure.Axes=None, show: bool=False,
			figsize: tuple=None, cmap: str='Spectral_r',
			sig_interactions: bool=True, title=None,
			):
	""" Heatmap visualising sender->receivers of cell type interactions.
		Parameters
		----------
		adata: AnnData
		use_label: str    Indicates the cell type labels or deconvolution results use for cell-cell interaction counting by LR pairs.
		lr: str    The LR pair to visualise the cci network for. If None, will use all pairs via adata.uns[f'lr_cci_{use_label}'].

		Returns
		-------
		ax: matplotlib.figure.Axes    Axes where the heatmap was drawn on if show=False.
	"""

	# Either plotting overall interactions, or just for a particular LR #
	int_df, title = get_int_df(adata, lr, use_label, sig_interactions, title)

	if type(figsize) == type(None): # Adjust size depending on no. cell types
		add = np.array([int_df.shape[0]*.1, int_df.shape[0]*.05])
		figsize = tuple(np.array([6.4, 4.8])+add)

	# Rank by total interactions #
	int_vals = int_df.values
	total_ints = int_vals.sum(axis=1)+int_vals.sum(axis=0)-int_vals.diagonal()
	order = np.argsort(-total_ints)
	int_df = int_df.iloc[order, order[::-1]]

	# Reformat the interaction df #
	flat_df = create_flat_df(int_df)

	ax = _box_map(flat_df['x'], flat_df['y'], flat_df['value'].astype(int),
				  ax=ax, figsize=figsize, cmap=cmap)

	ax.set_ylabel('Sender')
	ax.set_xlabel('Receiver')
	plt.suptitle(title)

	if show:
		plt.show()
	else:
		return ax

def lr_cci_map(adata: AnnData, use_label: str, lrs: list or np.array=None,
			   n_top_lrs: int=5, n_top_ccis: int=15, min_total: int=0,
			   ax: matplotlib.figure.Axes=None, figsize: tuple=(6.48,4.8),
			   show: bool=False, cmap: str='Spectral_r',
			   square_scaler: int=700, sig_interactions: bool=True):
	""" Heatmap of interaction counts; rows are lrs, columns are celltype->celltype interactions.
		Parameters
		----------
		adata: AnnData
		use_label: str    Indicates the cell type labels or deconvolution results use for cell-cell interaction counting by LR pairs.
		lrs: list-like    LR pairs to show in the heatmap, if None then top 5 lrs with highest no. of interactions used from adata.uns['lr_summary'].
		n_top_lrs: int    Indicates how many lrs to show; is ignored if lrs is not None.
		min_total: int    Minimum no. of totals interaction celltypes must have to be shown.
		square_scaler: int  Scaler to size the squares displayed.

		Returns
		-------
		ax: matplotlib.figure.Axes    Axes where the heatmap was drawn on if show=False.
	"""
	if sig_interactions:
		lr_int_dfs = adata.uns[f'per_lr_cci_{use_label}']
	else:
		lr_int_dfs = adata.uns[f'per_lr_cci_raw_{use_label}']

	if type(lrs)==type(None):
		lrs = np.array( list(lr_int_dfs.keys()) )
	else:
		lrs = np.array( lrs )
		n_top_lrs = len(lrs)

	# Creating a new int_df with lrs as rows & cell-cell as column #
	cell_types = list(lr_int_dfs.values())[0].index.values.astype(str)
	n_ints = len(cell_types)**2
	new_ints = np.zeros((len(lrs), n_ints))
	for lr_i, lr in enumerate(lrs):
		col_i = 0
		int_df = lr_int_dfs[lr]
		ccis = []
		for c_i, cell_i in enumerate(cell_types):
			for c_j, cell_j in enumerate(cell_types):
				new_ints[lr_i, col_i] = int_df.values[c_i, c_j]
				ccis.append( '->'.join([cell_i, cell_j]) )
				col_i += 1
	new_int_df = pd.DataFrame(new_ints, index=lrs, columns=ccis)

	# Filtering out ccis which have few LR interactions #
	total_ints = new_int_df.values.sum(axis=0)
	order = np.argsort(-total_ints)
	new_int_df = new_int_df.iloc[:, order[0:n_top_ccis]]

	# Getting the top_lrs to display by top loadings in PCA #
	if n_top_lrs < len(lrs):
		top_lrs = adata.uns['lr_summary'].index.values[0:n_top_lrs]
		new_int_df = new_int_df.loc[top_lrs,:]

	# Ordering by the no. of interactions #
	cci_ints = new_int_df.values.sum(axis=0)
	cci_order = np.argsort(-cci_ints)
	lr_ints = new_int_df.values.sum(axis=1)
	lr_order = np.argsort(-lr_ints)
	new_int_df = new_int_df.iloc[lr_order, cci_order]

	# Getting a flat version of the array for plotting #
	flat_df = create_flat_df(new_int_df.transpose())
	if flat_df.shape[0]==0 or flat_df.shape[1]==0:
		raise Exception(f'No interactions greater than min: {min_total}')

	ax = _box_map(flat_df['x'], flat_df['y'], flat_df['value'].astype(int),
				 ax=ax, cmap=cmap, figsize=figsize, square_scaler=square_scaler)

	ax.set_ylabel('LR-pair')
	ax.set_xlabel('Cell-cell interaction')

	if show:
		plt.show()
	else:
		return ax

def lr_chord_plot(adata: AnnData, use_label: str,
				  lr: str=None, min_ints: int=2, n_top_ccis: int=10,
				  cmap: str='default', show: bool=True,
				  sig_interactions: bool=True, title=None, label_size: int=10,
				  ):
	""" Chord diagram of interactions between cell types.
		Parameters
		----------
		adata: AnnData
		use_label: str    Indicates the cell type labels or deconvolution results use for cell-cell interaction counting by LR pairs.
		lr: str    The LR pair to visualise the cci network for. If None, will use all pairs via adata.uns[f'lr_cci_{use_label}'].
		min_ints: int    Minimum no. of interactions celltypes must have to be shown.

		Returns
		-------
		fig, ax: matplotlib.figure.Figure, matplotlib.figure.Axes   Axes where the heatmap was drawn on if show=False.
	"""
	# Either plotting overall interactions, or just for a particular LR #
	int_df, title = get_int_df(adata, lr, use_label, sig_interactions, title)

	int_df = int_df.transpose()
	fig = plt.figure(figsize=(8, 8))

	flux = int_df.values
	total_ints = flux.sum(axis=1) + flux.sum(axis=0) - flux.diagonal()
	keep = total_ints > min_ints
	# Limit of 10 for good display #
	if sum(keep) > n_top_ccis:
		keep = np.argsort(-total_ints)[0:n_top_ccis]
	flux = flux[:, keep]
	flux = flux[keep, :].astype(float)
	# Add pseudocount to row/column which has all zeros for the incoming
	# so can make the connection between the two
	for i in range(flux.shape[0]):
		if np.all(flux[i,:]==0):
			flux[i,flux[:,i]>0] += sys.float_info.min
		elif np.all(flux[:,i]==0):
			flux[flux[i, :] > 0, i] += sys.float_info.min

	cell_names = int_df.index.values.astype(str)[keep]
	nodes = cell_names

	# Retrieving colors of cell types #
	colors = get_colors(adata, use_label, cmap=cmap, label_set=cell_names)

	ax = plt.axes([0, 0, 1, 1])
	nodePos = chordDiagram(flux, ax, lim=1.25, colors=colors)
	ax.axis('off')
	prop = dict(fontsize=label_size, ha='center', va='center')
	for i in range(len(cell_names)):
		x, y = nodePos[i][0:2]
		ax.text(x, y, nodes[i],
				rotation=nodePos[i][2], size=10, **prop)
	fig.suptitle(title, fontsize=12, fontweight='bold')
	if show:
		plt.show()
	else:
		return fig, ax

""" Bokeh & grid plots; 
	has not been tested since multi-LR testing implimentation.
"""

def het_plot_interactive(adata: AnnData):
	bokeh_object = BokehCciPlot(adata)
	output_notebook()
	show(bokeh_object.app, notebook_handle=True)


def grid_plot(
	adata: AnnData,
	use_het: str = None,
	num_row: int = 10,
	num_col: int = 10,
	vmin: float = None,
	vmax: float = None,
	cropped: bool = True,
	margin: int = 100,
	dpi: int = 100,
	name: str = None,
	output: str = None,
	copy: bool = False,
) -> Optional[AnnData]:

	"""
	Cell diversity plot for sptial transcriptomics data.

	Parameters
	----------
	adata:                  Annotated data matrix.
	use_het:                Cluster heterogeneity count results from tl.cci.het
	num_row: int            Number of grids on height
	num_col: int            Number of grids on width
	cropped                 crop image or not.
	margin                  margin used in cropping.
	dpi:                    Set dpi as the resolution for the plot.
	name:                   Name of the output figure file.
	output:                 Save the figure as file or not.
	copy:                   Return a copy instead of writing to adata.

	Returns
	-------
	Nothing
	"""

	try:
		import seaborn as sns
	except:
		raise ImportError("Please run `pip install seaborn`")
	plt.subplots()

	sns.heatmap(
		pd.DataFrame(np.array(adata.obsm[use_het]).reshape(num_col, num_row)).T,
		vmin=vmin,
		vmax=vmax,
	)
	plt.axis("equal")

	if output is not None:
		plt.savefig(
			output + "/" + name + "_heatmap.pdf",
			dpi=dpi,
			bbox_inches="tight",
			pad_inches=0,
		)

	plt.show()
