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
   convert_scanpy
   create_stlearn


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
   add.labels
   add.annotation
   add.add_loupe_clusters
   add.add_mask
   add.apply_mask
   add.add_deconvolution


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

.. module:: stlearn.spatial.clustering
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   spatial.clustering.localization

.. module:: stlearn.spatial.trajectory
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   spatial.trajectory.pseudotime
   spatial.trajectory.pseudotimespace_global
   spatial.trajectory.pseudotimespace_local
   spatial.trajectory.compare_transitions
   spatial.trajectory.detect_transition_markers_clades
   spatial.trajectory.detect_transition_markers_branches
   spatial.trajectory.set_root

.. module:: stlearn.spatial.morphology
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   spatial.morphology.adjust

.. module:: stlearn.spatial.SME
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   spatial.SME.SME_impute0
   spatial.SME.pseudo_spot
   spatial.SME.SME_normalize

Tools: `tl`
-------------------

.. module:: stlearn.tl.clustering
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   tl.clustering.kmeans
   tl.clustering.louvain


.. module:: stlearn.tl.cci
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   tl.cci.load_lrs
   tl.cci.grid
   tl.cci.run
   tl.cci.adj_pvals
   tl.cci.run_lr_go
   tl.cci.run_cci

Plot: `pl`
-------------------

.. module:: stlearn.pl
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   pl.QC_plot
   pl.gene_plot
   pl.gene_plot_interactive
   pl.cluster_plot
   pl.cluster_plot_interactive
   pl.subcluster_plot
   pl.subcluster_plot
   pl.non_spatial_plot
   pl.deconvolution_plot
   pl.plot_mask
   pl.lr_summary
   pl.lr_diagnostics
   pl.lr_n_spots
   pl.lr_go
   pl.lr_result_plot
   pl.lr_plot
   pl.cci_check
   pl.ccinet_plot
   pl.lr_chord_plot
   pl.lr_cci_map
   pl.cci_map
   pl.lr_plot_interactive
   pl.spatialcci_plot_interactive

.. module:: stlearn.pl.trajectory
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   pl.trajectory.pseudotime_plot
   pl.trajectory.local_plot
   pl.trajectory.tree_plot
   pl.trajectory.transition_markers_plot
   pl.trajectory.DE_transition_plot

Tools: `datasets`
-------------------

.. module:: stlearn.datasets
.. currentmodule:: stlearn

.. autosummary::
   :toctree: .

   datasets.example_bcba()
