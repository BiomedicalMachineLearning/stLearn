[![Downloads](https://static.pepy.tech/personalized-badge/stlearn?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads)](https://pepy.tech/project/stlearn)

# stLearn - A downstream analysis toolkit for Spatial Transcriptomic data

**stLearn** is designed to comprehensively analyse Spatial Transcriptomics (ST) data to investigate complex biological processes within an undissociated tissue. ST is emerging as the “next generation” of single-cell RNA sequencing because it adds spatial and morphological context to the transcriptional profile of cells in an intact tissue section. However, existing ST analysis methods typically use the captured spatial and/or morphological data as a visualisation tool rather than as informative features for model development. We have developed an analysis method that exploits all three data types: Spatial distance, tissue Morphology, and gene Expression measurements (SME) from ST data. This combinatorial approach allows us to more accurately model underlying tissue biology, and allows researchers to address key questions in three major research areas: cell type identification, spatial trajectory reconstruction, and the study of cell-cell interactions within an undissociated tissue sample.

<p align="center">
<img src="https://i.imgur.com/yfXlCYO.png" alt="" width="386" height="261" />
</p>

## Detailed tutorials:

For installation and implementation of stLearn functionalities, see the dedicated documentation page at: https://stlearn.readthedocs.io/en/latest/

A collection of Jupyter notebooks are available in this GitHub repository in the **tutorials** folder 

## Brief installation instructions are below:

### For Linux/MacOS users

#### Step 1:

Prepare conda environment for stlearn

``` conda create -n stlearn python=3.8 ```

``` conda activate stlearn ```

#### Step 2:

``` conda config --add channels conda-forge ```

``` conda install jupyterlab louvain ipywidgets```


#### Step 3:

``` pip install stlearn```


### For Windows users

#### Step 1:

Prepare conda environment for stlearn

``` conda create -n stlearn python=3.8 ```

``` conda activate stlearn ```

#### Step 2:


``` conda install jupyterlab ipywidgets```

#### Step 3:

Access to: 

 - https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph

 - https://www.lfd.uci.edu/~gohlke/pythonlibs/#louvain-igraph

Download 2 files: python_igraph‑0.7.1.post6‑cp38‑cp38‑win_amd64.whl and louvain‑0.6.1‑cp38‑cp38‑win_amd64.whl

You have to change to downloaded files directory and install it:

``` pip install python_igraph‑0.7.1.post6‑cp38‑cp38‑win_amd64.whl ```

``` pip install louvain‑0.6.1‑cp38‑cp38‑win_amd64.whl ```


#### Step 4:

``` pip install stlearn```


### Popular bugs when install

- `DLL load failed while importing utilsextension: The specified module could not be found.`

You need to uninstall package `tables` and install it again


``` pip uninstall tables ```

``` pip install tables ```
