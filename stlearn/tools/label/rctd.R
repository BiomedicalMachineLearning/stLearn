# RCTD deconvolution wrapper script

library(RCTD)
library(data.table)

rctd <- function(st_counts, st_coords, sc_counts, sc_labels,
                 doublet_mode, min_cells, n_cores) {
  ###### Performs RCTD deconvolution of the st data from sc data #######

  # Making sure correct namings #
  colnames(st_counts) <- rownames(st_coords)

  # Subsetting to cell types with > X cells #
  label_set <- unique( sc_labels )
  label_counts <- as.integer( lapply(label_set,
                              function(label) {
                                length(which(sc_labels==label))
                              }))
  new_label_set <- label_set[ label_counts > min_cells ]
  labels_bool <- as.logical( lapply(sc_labels,
                             function(label) {
                               label %in% new_label_set
                             }))
  sc_counts <- sc_counts[,labels_bool]
  sc_labels <- as.factor( sc_labels[labels_bool] )
  names(sc_labels) <- colnames( sc_counts )

  ##############################################################################
                        # Creating RCTD objects #
  ##############################################################################
  ## Single cell reference ##
  reference <- Reference(sc_counts, cell_types = sc_labels)

  ## Spots for deconvolution ##
  query <- SpatialRNA(st_coords, st_counts)

  ## Creating the RCTD object ##
  myRCTD <- RCTD::create.RCTD(query, reference,
                              max_cores = n_cores, CELL_MIN_INSTANCE=min_cells)

  ##############################################################################
                        # Running RCTD #
  ##############################################################################
  myRCTD <- run.RCTD(myRCTD, doublet_mode = doublet_mode)

  ### Getting & normalising the results ###
  results <- myRCTD@results$weights
  norm_weights <- sweep(results, 1, rowSums(as.matrix(results)), '/')
  norm_weights <- as.data.frame(as.matrix(norm_weights))

  return( norm_weights )
}
