# Seurat label transfer wrapper script.
# Following from here: https://satijalab.org/seurat/articles/spatial_vignette.html
# See the section 'Integration with single cell data'

library(Seurat)
library(dplyr)

label_transfer <- function(st_expr_df, sc_expr_df, sc_labels) {
  ###### Performs Seurat label transfer from the sc data to the st data. #######

  # Creating the Seurat objects #
  print("Creating Seurat Objects.")
  st <- CreateSeuratObject(st_expr_df)
  VariableFeatures(st) <- rownames(st@assays$RNA@data)
  sc <- CreateSeuratObject(sc_expr_df)
  sc <- AddMetaData(sc, sc_labels, col.name='cell_type')
  VariableFeatures(sc) <- rownames(sc@assays$RNA@data)
  print("Finished creating Seurat.")

  # Finding variable features #
  #st <- FindVariableFeatures(st, selection.method="vst",
  #                                       nfeatures=n_highly_variable, verbose=T)
  #print("Finished finding variable features for ST data.")
  #sc <- FindVariableFeatures(sc, selection.method="vst",
  #                                       nfeatures=n_highly_variable, verbose=T)
  #print("Finished finding variable features for SC data.")

  # Dim reduction #
  st <- ScaleData(st, verbose=T)
  print("Finished scaling data for ST data.")
  st <- RunPCA(st, verbose = T, #features=rownames(st@assays$RNA@data)
              )
  print("Finished PCA for st data.")
  st <- RunUMAP(st, dims = 1:30, method='uwot-learn')
  print("Finished UMAP for st data.")

  sc <- ScaleData(sc) %>% RunPCA(verbose = T,
                                  #features=rownames(sc@assays$RNA@data)
                                  ) %>% RunUMAP(dims = 1:30,
                                                method='uwot-learn')
  print("Finished scaling data for SC data.")

  # Performing the label transfer #
  anchors <- FindTransferAnchors(reference = sc, query = st,
                                 #features=rownames(sc@assays$RNA@data),
                                 reference.reduction = 'pca', dims=1:30,
                                 normalization.method = "LogNormalize")
  print("Finished finding anchors.")
  predictions.assay <- TransferData(anchorset = anchors,
                                    refdata = sc$cell_type, prediction.assay=T,
                                    k.weight=20,
                                    weight.reduction = st[["pca"]], dims = 1:30)
  print("Finished label transferring.")
  transfer_scores <- predictions.assay@data

  return( as.data.frame( transfer_scores ) )
}
