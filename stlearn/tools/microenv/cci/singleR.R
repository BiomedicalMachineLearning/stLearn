# SingleR spot annotation wrapper script.

library(SingleR)

singleR <- function(st_expr_df, sc_expr_df, sc_labels,
                    n_centers, de_n, de_method) {

  st_expr <- as.double( as.data.frame(st_expr_df) )
  sc_expr <- as.double( as.data.frame(sc_expr_df) )

  ###### Performs Seurat label transfer from the sc data to the st data. #######
  sc_aggr <- aggregateReference(sc_expr, sc_labels, ncenters=n_centers)
  trained <- trainSingleR(sc_aggr, sc_aggr$label, de.n=de_n, de.method=de_method)

  out <- classifySingleR(st_expr, trained, fine.tune=F, prune=F)
  scores_df <- as.data.frame( out$scores )
  scores_df[,'labels'] <- out$labels

  return( scores_df )
}



