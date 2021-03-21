.. module:: stlearn
.. automodule:: stlearn
   :noindex:

API
======================================

Import stLearn as::

   import stlearn as st


Wrapper functions: `wrapper`
------------------------------

.. module:: stlearn.wrapper
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   Read10X
   ReadOldST
   ReadSlideSeq
   ReadMERFISH
   ReadSeqFish


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
   add.add_loupe_clusters
   add.auto_annotate


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
   spatial.trajectory.pseudotime
   spatial.trajectory.pseudotimespace_global
   spatial.trajectory.pseudotimespace_local
   spatial.trajectory.compare_transitions
   spatial.trajectory.detect_transition_markers_clades
   spatial.trajectory.detect_transition_markers_branches
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
   pl.deconvolution_plot
   pl.QC_plot
   pl.het_plot
   pl.trajectory.pseudotime_plot
   pl.trajectory.local_plot
   pl.trajectory.tree_plot
   pl.trajectory.transition_markers_plot
   pl.trajectory.DE_transition_plot
