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

	pip install -U stlearn




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

 - https://www.lfd.uci.edu/~gohlke/pythonlibs/#leidenalg

Download 3 files: igraph‑0.9.9‑cp38‑cp38‑win_amd64.whl, louvain‑0.7.1‑cp38‑cp38‑win_amd64.whl, leidenalg‑0.8.8‑cp38‑cp38‑win_amd64.whl

You have to change to downloaded files directory and install those packages.

**Step 4:**

If you have previous version of stLearn, please uninstall it.

::

	pip uninstall stlearn

**Step 5:**
::

	pip install -U stlearn

Popular bugs
---------------

- `DLL load failed while importing utilsextension: The specified module could not be found.`

You need to uninstall package `tables` and install it again
::

	pip uninstall tables
	pip install tables
