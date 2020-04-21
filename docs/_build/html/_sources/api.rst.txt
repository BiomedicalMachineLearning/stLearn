.. module:: stlearn
.. automodule:: stlearn
   :noindex:

API
======================================

Import stLearn as::

   import stlearn as st




Reading: `read`
-------------------

.. module:: stlearn.read
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   read.file_table
   read.file_10x_mtx
   read.file_10x_h5



Add: `add`
-------------------

.. module:: stlearn.add
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   add.image
   add.positions
   add.parsing
   add.lr
   add.annotation
   add.cpdb


Preprocessing: `pp`
-------------------

.. module:: stlearn.pp
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   pp.filter_genes
   pp.log1p
   pp.normalize_total
   pp.scale
   pp.neighbors
   pp.tiling
   pp.extract_feature



Embedding: `em`
-------------------

.. module:: stlearn.em
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   em.run_pca
   em.run_umap
   em.run_ica
   em.run_fa
   em.run_diffmap


Spatial: `spatial`
-------------------

.. module:: stlearn.spatial
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   spatial.clustering.localization
   spatial.trajectory.global_level
   spatial.trajectory.local_level
   spatial.morphology.adjust

Tools: `tl`
-------------------

.. module:: stlearn.tl
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   tl.clustering.kmeans
   tl.clustering.louvain
   tl.cci.lr
   tl.cci.merge
   tl.SpatialDE


Plot: `pl`
-------------------

.. module:: stlearn.pl
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   pl.gene_plot
   pl.cluster_plot
   pl.subcluster_plot
   pl.microenv_plot
   pl.non_spatial_plot
   pl.QC_plot
   pl.het_plot
   pl.violin_plot
   pl.stacked_bar_plot
   pl.trajectory.global_plot
   pl.trajectory.local_plot
   pl.trajectory.tree_plot

.. note::
   Wrappers to external functionality are found in :mod:`stlearn.external`.