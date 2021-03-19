stLearn - a downstream analysis toolkit for Spatial Transcriptomics data
============================================================================

.. image:: https://i.imgur.com/yfXlCYO.png
   :width: 300px
   :align: left

stLearn is designed to comprehensively analyse Spatial Transcriptomics (ST) data to investigate complex biological processes within an undissociated tissue. ST is emerging as the “next generation” of single-cell RNA sequencing because it adds spatial and morphological context to the transcriptional profile of cells in an intact tissue section. However, existing ST analysis methods typically use the captured spatial and/or morphological data as a visualisation tool rather than as informative features for model development. We have developed an analysis method that exploits all three data types: Spatial distance, tissue Morphology, and gene Expression measurements (SME) from ST data. This combinatorial approach allows us to more accurately model underlying tissue biology, and allows researchers to address key questions in three major research areas: cell type identification, cell trajectory reconstruction, and the study of cell-cell interactions within an undissociated tissue sample.

We also published stLearn-interactive which is a python-based interactive website for working with all the functions from stLearn and upgrade with some bokeh-based plots.

stLearn-interactive source code and installation: `Github <https://github.com/BiomedicalMachineLearning/stlearn_interactive>`_

stLearn-interactive tutorial: `Wiki page <https://github.com/BiomedicalMachineLearning/stlearn_interactive/wiki/stLearn-interactive-tutorial>`_



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

   stlearn-interactive-tutorial
   stSME_clustering
   stSME_comparison
   Pseudo-time-space-tutorial
   stLearn-CCI
   Working-with-Old-Spatial-Transcriptomics-data
   Read_slideseq
   Read_MERFISH
   Read_seqfish
   ST_deconvolution_visualization
