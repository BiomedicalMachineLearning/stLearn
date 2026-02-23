.. module:: stlearn
.. automodule:: stlearn
   :noindex:

API
======================================

Import stLearn as::

   import stlearn as st


Wrapper functions: `wrapper`
------------------------------

.. currentmodule:: stlearn

.. autosummary::
   :toctree: api/

   read_10x
   read_old_st
   read_slide_seq
   read_merfish
   read_seq_fish
   read_xenium
   create_stlearn
   convert_scanpy


Add: `add`
-------------------

.. currentmodule:: stlearn

.. autosummary::
   :toctree: api/

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
   :toctree: api/

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
   :toctree: api/

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
   :toctree: api/

   spatial.clustering.localization

.. module:: stlearn.spatial.trajectory
.. currentmodule:: stlearn

.. autosummary::
   :toctree: api/

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
   :toctree: api/

   spatial.morphology.adjust

.. module:: stlearn.spatial.sme
.. currentmodule:: stlearn

.. autosummary::
   :toctree: api/

   spatial.sme.sme_impute0
   spatial.sme.pseudo_spot
   spatial.sme.sme_normalize

Tools: `tl`
-------------------

.. currentmodule:: stlearn

.. autosummary::
   :toctree: api/

   tl.clustering.kmeans
   tl.clustering.leiden
   tl.cci.load_lrs
   tl.cci.grid
   tl.cci.run
   tl.cci.adj_pvals
   tl.cci.run_lr_go
   tl.cci.run_cci

Plot: `pl`
-------------------

.. currentmodule:: stlearn

.. autosummary::
   :toctree: api/

   pl.qc_plot
   pl.gene_plot
   pl.gene_plot_interactive
   pl.cluster_plot
   pl.cluster_plot_interactive
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

.. currentmodule:: stlearn

.. autosummary::
   :toctree: api/

   pl.trajectory.pseudotime_plot
   pl.trajectory.local_plot
   pl.trajectory.tree_plot
   pl.trajectory.transition_markers_plot
   pl.trajectory.de_transition_plot

Datasets: `datasets`
---------------------------

.. currentmodule:: stlearn

.. autosummary::
   :toctree: api/

   datasets.visium_sge
   datasets.xenium_sge
