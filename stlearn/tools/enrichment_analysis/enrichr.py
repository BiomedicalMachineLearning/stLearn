from anndata import AnnData
from typing import Optional, Union
from scipy.stats import pearsonr
import gseapy
import pandas as pd


def enrichr(
    adata: AnnData,
    use_data: str = "X_fa",
    factor: int = 1,
    top_genes: list = None,
    gene_sets: str = "KEGG_2019_Human",
    cutoff: float = 0.05,
    copy: bool = False
) -> Optional[AnnData]:
    
    print("Check more gene sets here: https://amp.pharm.mssm.edu/Enrichr/#stats")

    if top_genes is None:
    
        top_genes = adata.uns["factor_sig"][use_data]["Factor_" + str(factor)]["top_genes"]

        gl = gseapy.enrichr(gene_list=top_genes,
                            gene_sets=gene_sets,
                            cutoff=cutoff, outdir=None)

        result = gl.results

        result = result[result["Adjusted P-value"] < cutoff]

        adata.uns['factor_sig'][use_data]["Factor_"+str(factor)][gene_sets] = {}

        adata.uns['factor_sig'][use_data]["Factor_"+str(factor)][gene_sets].update({"result":result}) 

        print('Enrichment analysis of top genes in Factor ' + str(factor) + 
            ' is done! The result is stored in adata.uns["factor_sig"]["' + use_data+ '"]'+'["Factor_'+str(factor)+'"]["'+gene_sets+'"]["result"]')

    else:
        gl = gseapy.enrichr(gene_list=top_genes,
                            gene_sets=gene_sets,
                            cutoff=cutoff, outdir=None)

        result = gl.results

        result = result[result["Adjusted P-value"] < cutoff]
        
        
        adata.uns['enrichr_'+ gene_sets] = result


        print('Enrichment analysis of top genes is done! The result is stored in adata.uns["enrichr_'+gene_sets+ '""]')


    return adata if copy else None