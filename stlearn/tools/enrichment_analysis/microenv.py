from anndata import AnnData
from typing import Optional, Union
from scipy.stats import pearsonr
import gseapy
import pandas as pd


def microenv(
    adata: AnnData,
    use_data: str = "X_fa",
    factor: int = 1,
    gene_sets: str = "KEGG_2019_Human",
    cutoff: float = 0.05,
    n_top: int = 100,
    copy: bool = False,
) -> Optional[AnnData]:

    ft = adata.obsm[use_data][:,factor]

    genes = []
    result_pearson = []
    for gene in adata.var_names:
        
        genes.append(gene)
        result_pearson.append(pearsonr(ft,adata[:, gene].X.toarray().reshape(-1,len(adata))[0])[0])

    cor_results = pd.DataFrame(
        {'Gene': genes,
         'Pearson_correlation': result_pearson,
        })

    top_genes = list(cor_results.sort_values("Pearson_correlation",ascending=False)[:n_top].Gene)

    if "factor_sig" not in adata.uns:
            adata.uns['factor_sig'] = {}

    adata.uns['factor_sig'].update({use_data: {}})

    adata.uns['factor_sig'][use_data].update({"Factor_"+str(factor):{}})

    adata.uns['factor_sig'][use_data]["Factor_"+str(factor)].update({"top_genes" :top_genes}) 

    print('Get top genes of Factor ' + str(factor) + 
        ' is done! The top genes are stored in adata.uns["factor_sig"]["' + use_data+ '"]'+'["Factor_'+str(factor)+'"]["result"]')



    params = {'params':{'n_top': n_top,'factor': factor}}

    adata.uns['factor_sig'][use_data]["Factor_"+str(factor)].update({"params":params}) 



    return adata if copy else None