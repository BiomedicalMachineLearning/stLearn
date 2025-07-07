# R script that runs the LR GO analysis #

library(clusterProfiler)
library(org.Mm.eg.db)
library(org.Hs.eg.db)
#library(enrichplot)
#library(ggplot2)

GO_analyse <- function(genes, bg_genes, species,
                       p_cutoff, q_cutoff, onts) {
                       #p_cutoff=.01, q_cutoff=0.5, onts='BP') {
  # Main function for performing the GO analysis #

  # Selecting correct species database #
  if (species == 'human') {db <- org.Hs.eg.db
  } else {db <- org.Mm.eg.db}

  # Performing the enrichment #
  em <- enrichGO(genes, db, ont=onts, keyType='SYMBOL',
                 pvalueCutoff=p_cutoff, qvalueCutoff=q_cutoff, universe=bg_genes)

  result <- em@result
  sig_results <- result[,'p.adjust']<p_cutoff
  result <- result[sig_results,]

  #### Couldn't get the visualising functioning in reasonable way ####
  # Creating the summary plot #
  #edo <- pairwise_termsim(em)
  #p <- emapplot(edo, showCategory=20,
  #              pie_scale=1, repel=T,
  #              cex_label_category=1.2, layout='nicely', color='p.adjust',
  #)+theme(text=element_text(face='bold', size=1))
  #p$layers[[3]]$aes_params[['fontface']] <- "bold"

  #all_results <- list('go_df'= results, 'go_vis' = p)
  return( result )
}
