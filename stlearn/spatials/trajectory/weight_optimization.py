import numpy as np
import networkx as nx
from .global_level import global_level
from .local_level import local_level
from .utils import lambda_dist,resistance_distance


def weight_optimizing_global(adata,use_label=None,list_cluster=None,step=0.01,k=10):
	# Screening PTS graph
	print("Screening PTS graph...")
	Gs = []
	j = 0
	for i in range(0,int(1/step + 1)):

	    Gs.append(nx.adjacency_matrix(global_level(adata,use_label=use_label,list_cluster=list_cluster,w=round(j,2),return_graph=True)))

	    j = j+step

	# Calculate the graph dissimilarity using Laplacian matrix
	print("Calculate the graph dissimilarity using Laplacian matrix...")
	result = []
	a1_list = []
	a2_list = []
	indx = []
	w = 0
	for i in range(1,int(1/step)):
	    w+=step
	    a1 = lambda_dist(Gs[i],Gs[0],k=k)
	    a2 = lambda_dist(Gs[i],Gs[-1],k=k)
	    a1_list.append(a1)
	    a2_list.append(a2)
	    indx.append(w)
	    result.append(np.absolute(1-a1/a2))
	

	def NormalizeData(data):
	    return (data - np.min(data)) / (np.max(data) - np.min(data))

	result = NormalizeData(result)
	    
	optimized_ind = np.where(result == np.amin(result))[0][0]
	opt_w = round(indx[optimized_ind],2)
	print("The optimized weighting is:", str(opt_w))

	return opt_w


def weight_optimizing_local(adata,use_label=None,cluster=None,step=0.01):
	# Screening PTS graph
	print("Screening PTS graph...")
	Gs = []
	j = 0
	for i in range(0,int(1/step + 1)):

	    Gs.append(local_level(adata,use_label=use_label,cluster=cluster,w=round(j,2),verbose=False,return_matrix=True))

	    j = j+step

	# Calculate the graph dissimilarity using Laplacian matrix
	print("Calculate the graph dissimilarity using Resistance distance...")
	result = []
	a1_list = []
	a2_list = []
	indx = []
	w = 0
	for i in range(1,int(1/step)):
	    w+=step
	    a1 = resistance_distance(Gs[i],Gs[0])
	    a2 = resistance_distance(Gs[i],Gs[-1])
	    a1_list.append(a1)
	    a2_list.append(a2)
	    indx.append(w)
	    result.append(np.absolute(1-a1/a2))
	

	def NormalizeData(data):
	    return (data - np.min(data)) / (np.max(data) - np.min(data))

	result = NormalizeData(result)
	    
	optimized_ind = np.where(result == np.amin(result))[0][0]
	opt_w = round(indx[optimized_ind],2)
	print("The optimized weighting is:", str(opt_w))

	return opt_w