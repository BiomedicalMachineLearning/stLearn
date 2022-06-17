.. highlight:: shell

============
Installation
============


Install by Anaconda
---------------

**Step 1:**

Prepare conda environment for stLearn
::

	conda create -n stlearn python=3.8
	conda activate stlearn

**Step 2:**

You can directly install stlearn in the anaconda by:
::

	conda install -c conda-forge stlearn

Install by PyPi
---------------

**Step 1:**

Prepare conda environment for stLearn
::

	conda create -n stlearn python=3.8
	conda activate stlearn

**Step 2:**

Install stlearn using `pip`
::

	pip install -U stlearn



Popular bugs
---------------

- `DLL load failed while importing utilsextension: The specified module could not be found.`

You need to uninstall package `tables` and install it again
::

	pip uninstall tables
	conda install pytables

If conda version does not work, you can access to this site and download the .whl file: `https://www.lfd.uci.edu/~gohlke/pythonlibs/#pytables`

::

	pip install tables-3.7.0-cp38-cp38-win_amd64.whl
