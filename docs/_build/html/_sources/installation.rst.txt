.. highlight:: shell

============
Installation
============



For Linux/MacOS users
-----------------------

**Step 1:**

Prepare conda environment for stLearn
::

	conda create -n stlearn python=3.8
	conda activate stlearn

**Step 2:**

::

	conda config --add channels conda-forge
	conda install jupyterlab louvain ipywidgets

**Step 3:**
::

	pip install stlearn




For Windows users
-----------------------

**Step 1:**

Prepare conda environment for stLearn
::

	conda create -n stlearn python=3.8
	conda activate stlearn

**Step 2:**

::

	conda install jupyterlab ipywidgets

**Step 3:**

Access to: 

 - https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph

 - https://www.lfd.uci.edu/~gohlke/pythonlibs/#louvain-igraph

Download 2 files: python_igraph‑0.7.1.post6‑cp37‑cp37m‑win_amd64.whl and louvain‑0.6.1‑cp37‑cp37m‑win_amd64.whl

You have to change to downloaded files directory and install it:
::

	pip install python_igraph‑0.7.1.post6‑cp37‑cp37m‑win_amd64.whl
	pip install louvain‑0.6.1‑cp37‑cp37m‑win_amd64.whl

**Step 4:**
::

	pip install stlearn

Popular bugs
---------------

- `DLL load failed while importing utilsextension: The specified module could not be found.`

You need to uninstall package `tables` and install it again
::

	pip uninstall tables
	pip install tables