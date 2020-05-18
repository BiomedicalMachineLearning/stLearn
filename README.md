# stLearn - A downstream analysis toolkit for Spatial Transcriptomic data (v0.1.7)

**stLearn** is designed to comprehensively analyse Spatial Transcriptomics (ST) data to investigate complex biological processes within an undissociated tissue. ST is emerging as the “next generation” of single-cell RNA sequencing because it adds spatial and morphological context to the transcriptional profile of cells in an intact tissue section. However, existing ST analysis methods typically use the captured spatial and/or morphological data as a visualisation tool rather than as informative features for model development. We have developed an analysis method that exploits all three data types: Spatial distance, tissue Morphology, and gene Expression measurements (SME) from ST data. This combinatorial approach allows us to more accurately model underlying tissue biology, and allows researchers to address key questions in three major research areas: cell type identification, cell trajectory reconstruction, and the study of cell-cell interactions within an undissociated tissue sample.

<img src="https://i.imgur.com/yfXlCYO.png" alt="" width="400" height="400" />

## Detailed tutorials:

For installation and implementation of stLearn functionalities, see the dedicated documentation page at: https://stlearn.readthedocs.io/en/latest/

A collection of Jupyter notebooks are available in this repository in the **tutorial** folder 

## Brief installation instructions are below:


### Step 0:

Prepare conda environment for stlearn

``` conda create -n stlearn python=3.8 ```

### Step 1:

``` conda config --add channels conda-forge ```

``` conda install jupyterlab louvain ipywidgets```
### Step 2 (For Windows user):

Access to: 

 - https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph

 - https://www.lfd.uci.edu/~gohlke/pythonlibs/#louvain-igraph

Download 2 files: python_igraph‑0.7.1.post6‑cp37‑cp37m‑win_amd64.whl and louvain‑0.6.1‑cp37‑cp37m‑win_amd64.whl

You have to change to downloaded files directory and install it:

``` pip install python_igraph‑0.7.1.post6‑cp37‑cp37m‑win_amd64.whl ```

``` pip install louvain‑0.6.1‑cp37‑cp37m‑win_amd64.whl ```



### Step 3:

``` pip install stlearn```




