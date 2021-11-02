# Seurat label transfer wrapper script.
# Following from here: https://satijalab.org/seurat/articles/spatial_vignette.html
# See the section 'Integration with single cell data'

library(Seurat)
library(dplyr)

label_transfer <- function(st_expr_df, sc_expr_df, sc_labels) {
  ###### Performs Seurat label transfer from the sc data to the st data. #######

  # Creating the Seurat objects #
  st <- CreateSeuratObject(st_expr_df)
  sc <- CreateSeuratObject(sc_expr_df)
  sc <- AddMetaData(sc, sc_labels, col.name='cell_type')

  # Finding variable features #
  st <- FindVariableFeatures(st, selection.method="vst", nfeatures=2000,
                                                                      verbose=F)
  sc <- FindVariableFeatures(sc, selection.method="vst", nfeatures=2000,
                                                                      verbose=F)

  # Dim reduction #
  st <- ScaleData(st) %>% RunPCA(verbose = FALSE) %>% RunUMAP(dims = 1:30)
  sc <- ScaleData(sc) %>% RunPCA(verbose = FALSE) %>% RunUMAP(dims = 1:30)

  # Performing the label transfer #
  anchors <- FindTransferAnchors(reference = sc, query = st,
                                 reference.reduction = 'pca', dims=1:30,
                                 normalization.method = "LogNormalize")
  predictions.assay <- TransferData(anchorset = anchors,
                                    refdata = sc$cell_type, prediction.assay=T,
                                    weight.reduction = st[["pca"]], dims = 1:30)
  transfer_scores <- predictions.assay@data

  return( transfer_scores )
}






