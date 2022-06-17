<p align="center">
  <img src="https://i.imgur.com/yfXlCYO.png"
    alt="deepreg_logo" title="DeepReg" width="300"/>
</p>

<table align="center">
  <tr>
    <td>
      <b>Package</b>
    </td>
    <td>
      <a href="https://pypi.python.org/pypi/stlearn/">
      <img src="https://img.shields.io/pypi/v/stlearn.svg" alt="PyPI Version">
      </a>
      <a href="https://pepy.tech/project/stlearn">
      <img src="https://static.pepy.tech/personalized-badge/stlearn?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads"
        alt="PyPI downloads">
      </a>
      <a href="https://anaconda.org/conda-forge/stlearn">
      <img src="https://anaconda.org/conda-forge/stlearn/badges/downloads.svg" alt="Conda downloads">
      </a>
      <a href="https://anaconda.org/conda-forge/stlearn">
      <img src="https://anaconda.org/conda-forge/stlearn/badges/installer/conda.svg" alt="Install">
      </a>
    </td>
  </tr>
  <tr>
    <td>
      <b>Documentation</b>
    </td>
    <td>
      <a href="https://stlearn.readthedocs.io/en/latest/">
      <img src="https://readthedocs.org/projects/stlearn/badge/?version=latest" alt="Documentation Status">
      </a>
    </td>
  </tr>
  <tr>
    <td>
     <b>Paper</b>
    </td>
    <td>
      <a href="https://doi.org/10.1101/2020.05.31.125658"><img src="https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg"
        alt="DOI"></a>
    </td>
  </tr>
  <tr>
    <td>
      <b>License</b>
    </td>
    <td>
      <a href="https://github.com/BiomedicalMachineLearning/stLearn/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-BSD-blue.svg"
        alt="LICENSE"></a>
    </td>
  </tr>
</table>


# stLearn - A downstream analysis toolkit for Spatial Transcriptomic data

**stLearn** is designed to comprehensively analyse Spatial Transcriptomics (ST) data to investigate complex biological processes within an undissociated tissue. ST is emerging as the “next generation” of single-cell RNA sequencing because it adds spatial and morphological context to the transcriptional profile of cells in an intact tissue section. However, existing ST analysis methods typically use the captured spatial and/or morphological data as a visualisation tool rather than as informative features for model development. We have developed an analysis method that exploits all three data types: Spatial distance, tissue Morphology, and gene Expression measurements (SME) from ST data. This combinatorial approach allows us to more accurately model underlying tissue biology, and allows researchers to address key questions in three major research areas: cell type identification, spatial trajectory reconstruction, and the study of cell-cell interactions within an undissociated tissue sample.

---

## Getting Started

- [Documentation and Tutorials](https://stlearn.readthedocs.io/en/latest/)

## New features

`stlearn.pl.gene_plot_interactive`

<img src="https://media.giphy.com/media/hUHAZcbVMm5pdUKMq4/giphy.gif" width="600" height="432" />

## Citing stLearn

If you have used stLearn in your research, please consider citing us:

> Pham _et al._, (2020). stLearn: integrating spatial location, tissue morphology and gene expression to find cell types, cell-cell interactions and spatial trajectories within undissociated tissues
> _biorxiv_
> https://doi.org/10.1101/2020.05.31.125658
