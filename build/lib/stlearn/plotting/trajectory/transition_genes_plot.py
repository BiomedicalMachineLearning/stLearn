import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from decimal import Decimal
from stlearn.spatials.trajectory.extra import adjust_spines
from anndata import AnnData
from typing import Optional, Union

def transition_genes_plot(
    adata: AnnData,
    subcluster: int = 0,
    num_genes: int = 15,
    cmap: str = "Set2",
    dpi: int = 96,
    ) -> Optional[AnnData]:

    plt.rcParams['figure.dpi'] = dpi

    subcl_adata = adata.uns["subcluster_" + str(subcluster) +"_adata"]

    dict_tg_edges = subcl_adata.uns['transition_genes']
    flat_tree = subcl_adata.uns["pseudotimespace"]['flat_tree']
    # dict_node_state = nx.get_node_attributes(flat_tree,'label')    
    colors = sns.color_palette(cmap, n_colors=8, desat=0.8)
    for edge_i in dict_tg_edges.keys():

        df_tg_edge_i = deepcopy(dict_tg_edges[edge_i])
        df_tg_edge_i = df_tg_edge_i.iloc[:num_genes,:]

        stat = df_tg_edge_i.stat[::-1]
        qvals = df_tg_edge_i.qval[::-1]

        pos = np.arange(df_tg_edge_i.shape[0])-1
        bar_colors = np.tile(colors[4],(len(stat),1))
        # bar_colors = repeat(colors[0],len(stat))
        id_neg = np.arange(len(stat))[np.array(stat<0)]
        bar_colors[id_neg]=colors[2]

        fig = plt.figure(figsize=(12,np.ceil(0.4*len(stat))))
        ax = fig.add_subplot(1,1,1, adjustable='box')
        ax.barh(pos,stat,align='center',height=0.8,tick_label=[''],color = bar_colors)
        ax.set_xlabel('Spearman Correlation Coefficient')
        ax.set_title("branch " + edge_i[0]+'_'+edge_i[1])

        adjust_spines(ax, ['bottom'])
        ax.spines['left'].set_position('center')
        ax.spines['left'].set_color('none')
        ax.set_xlim((-1,1))
        ax.set_ylim((min(pos)-1,max(pos)+1))

        rects = ax.patches
        for i,rect in enumerate(rects):
            if(stat[i]>0):
                alignment = {'horizontalalignment': 'left', 'verticalalignment': 'center'}
                ax.text(rect.get_x()+rect.get_width()+0.02, rect.get_y() + rect.get_height()/2.0, \
                        qvals.index[i],fontsize=12,**alignment)
                ax.text(rect.get_x()+0.02, rect.get_y()+rect.get_height()/2.0, \
                        "{:.2E}".format(Decimal(str(qvals[i]))),color='black',fontsize=9,**alignment)
            else:
                alignment = {'horizontalalignment': 'right', 'verticalalignment': 'center'}
                ax.text(rect.get_x()+rect.get_width()-0.02, rect.get_y()+rect.get_height()/2.0, \
                        qvals.index[i],fontsize=12,**alignment)
                ax.text(rect.get_x()-0.02, rect.get_y()+rect.get_height()/2.0, \
                        "{:.2E}".format(Decimal(str(qvals[i]))),color='w',fontsize=9,**alignment)
