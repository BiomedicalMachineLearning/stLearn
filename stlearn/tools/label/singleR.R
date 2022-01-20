# SingleR spot annotation wrapper script.

library(SingleR)

singleR <- function(st_expr_df, sc_expr_df, sc_labels,
                    n_centers, de_n, de_method) {
  ##### Runs SingleR spot annotation #######
  st_expr <- as.matrix( st_expr_df )
  sc_expr <- as.matrix( sc_expr_df )

  ##### Subsetting to genes in common between datasets #####
  common_genes <- intersect(rownames(sc_expr), rownames(st_expr))

  sc_expr <- sc_expr[common_genes,]
  st_expr <- st_expr[common_genes,]

  ###### Performs Seurat label transfer from the sc data to the st data. #######
  sc_aggr <- aggregateReference(sc_expr, sc_labels, ncenters=n_centers)
  trained <- trainSingleR(sc_aggr, sc_aggr$label, de.n=de_n, de.method=de_method)

  out <- classifySingleR(st_expr, trained, fine.tune=F, prune=F)
  scores_df <- as.data.frame( out$scores )
  scores_df[,'labels'] <- out$labels
  rownames(scores_df) <- out@rownames

  return( scores_df )
}
