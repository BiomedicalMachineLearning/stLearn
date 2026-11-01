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

   convert_scanpy
   concatenate_spatial_adata


Reading and writing: `io`
------------------------------

.. module:: stlearn.io
.. currentmodule:: stlearn

.. autosummary::
   :toctree: api/

   io.read_10x
   io.read_xenium
   io.read_old_st
   io.read_slide_seq
   io.read_merfish
   io.read_seq_fish
   io.create_stlearn

.. autosummary::
   :toctree: api/

   io.visium
   io.visium_hd
   io.xenium
   io.attach_spatial
   io.attach_spatial_sdata
   io.visium_scalefactors
   io.xenium_pixel_size
   io.crop_to_cells
   io.crop_visium_hd
   io.crop_xenium
   io.save_table
   io.checkpoint
   io.store_checkpoint


Add: `add`
-------------------

.. currentmodule:: stlearn

.. autosummary::
   :toctree: api/

   add.image
   add.positions
   add.parsing
   add.labels
   add.annotation
   add.row_annotations
   add.polygon_annotations
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
   pp.filter_cells
   pp.log1p
   pp.normalize_total
   pp.highly_variable_genes
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
   tl.clustering.annotate_interactive
   tl.label.run_label_transfer
   tl.label.run_rctd
   tl.label.run_single_r

.. autosummary::
   :toctree: api/

   tl.cci.lr_pairs
   tl.cci.read_lr_pairs
   tl.cci.LRPairs
   tl.cci.run
   tl.cci.adj_pvals
   tl.cci.lr_result
   tl.cci.run_lr_go
   tl.cci.run_cci
   tl.cci.grid
   tl.cci.create_grid
   tl.cci.load_lrs

Plot: `pl`
-------------------

.. currentmodule:: stlearn

.. autosummary::
   :toctree: api/

   pl.qc_plot
   pl.gene_plot
   pl.gene_plot_interactive
   pl.feat_plot
   pl.cluster_plot
   pl.cluster_plot_interactive
   pl.subcluster_plot
   pl.non_spatial_plot
   pl.stack_3d_plot
   pl.sdata_plot
   pl.deconvolution_plot
   pl.plot_mask
   pl.lr_summary
   pl.lr_diagnostics
   pl.lr_n_spots
   pl.lr_go
   pl.lr_result_plot
   pl.lr_plot
   pl.het_plot
   pl.grid_plot
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
   pl.trajectory.check_trajectory

Datasets: `datasets`
---------------------------

.. currentmodule:: stlearn

.. autosummary::
   :toctree: api/

   datasets.visium_sge
   datasets.visium_hd_sge
   datasets.xenium_sge
   datasets.atera_sge