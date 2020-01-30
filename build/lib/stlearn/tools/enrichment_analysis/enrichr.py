from anndata import AnnData
from typing import Optional, Union
from scipy.stats import pearsonr
import gseapy
import pandas as pd


def enrichr(
    adata: AnnData,
    module: str = "env",
    method: str = "fa",
    factor: int = 1,
    gene_sets: str = "KEGG_2019_Human",
    cutoff: float = 0.05,
    copy: bool = False,
) -> Optional[AnnData]:
    
    print("Check more gene sets here: https://amp.pharm.mssm.edu/Enrichr/#stats")

    if module == "env":
    
        top_genes = adata.uns["factor_sig"][method]["Factor_" + str(factor)]["top_genes"]

        gl = gseapy.enrichr(gene_list=top_genes,
                            gene_sets=gene_sets,
                            cutoff=cutoff, outdir=None)

        result = gl.results

        result = result[result["Adjusted P-value"] < cutoff]

        adata.uns['factor_sig'][method]["Factor_"+str(factor)][gene_sets] = {}

        adata.uns['factor_sig'][method]["Factor_"+str(factor)][gene_sets].update({"result":result}) 

        print('Enrichment analysis of top genes in Factor ' + str(factor) + 
            ' is done! The result is stored in adata.uns["factor_sig"]["' + method+ '"]'+'["Factor_'+str(factor)+'"]["'+gene_sets+'"]["result"]')

    


    return adata if copy else None