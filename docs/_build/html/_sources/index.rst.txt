stLearn - a downstream analysis toolkit for Spatial Transcriptomics data
============================================================================

.. image:: https://i.imgur.com/yfXlCYO.png
   :width: 300px
   :align: left
   
stLearn is designed to comprehensively analyse Spatial Transcriptomics (ST) data to investigate complex biological processes within an undissociated tissue. ST is emerging as the “next generation” of single-cell RNA sequencing because it adds spatial and morphological context to the transcriptional profile of cells in an intact tissue section. However, existing ST analysis methods typically use the captured spatial and/or morphological data as a visualisation tool rather than as informative features for model development. We have developed an analysis method that exploits all three data types: Spatial distance, tissue Morphology, and gene Expression measurements (SME) from ST data. This combinatorial approach allows us to more accurately model underlying tissue biology, and allows researchers to address key questions in three major research areas: cell type identification, cell trajectory reconstruction, and the study of cell-cell interactions within an undissociated tissue sample. 

.. toctree::
   :maxdepth: 1
   :caption: Content:

   installation
   api
   authors
   history

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   stSME_clustering
   stSME_comparison
   Pseudo-space-time-tutorial
   cci
   cci_within-spot_single_lr
   cci_within-spot_cpdb_lrs
   cci_between-spot_single_lr
   cci_between-spot_cpdb_lrs
   Working-with-Old-Spatial-Transcriptomics-data
   Read_slideseq
   Read_MERFISH
   Read_seqfish
   ST_deconvolution_visualization
