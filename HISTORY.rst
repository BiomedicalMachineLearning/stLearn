=======
History
=======

1.5.1 (2026-11-09)
------------------
* ``st.tl.cci.run`` now uses a spots-by-spots, sparse graph:
  ``adata.obsp["spot_neighbours"]``. It replaces the comma-joined strings
  ``adata.obsm["spot_neighbours"]`` and ``adata.obsm["spot_neigh_bcs"]``.
  ``st.tl.cci.run_cci`` and ``get_edges`` read the stored graph directly.
* With ``use_label``, ``run`` counts cell types over the same graph instead
  of searching the neighbours a second time (``het.count`` takes it as
  ``neighbours=``).

API and Bug Fixes:

* ``st.tl.cci.run`` no longer stores ``lrfeatures`` in the LR table's ``uns``.
  ``nonzero-median``, ``zero-prop``, ``median_rank``, ``prop_rank`` and
  ``mean_rank`` are columns of ``lr_results.var``.
  ``lr_results.varm["lr_quantiles"]`` stores ligand and receptor expression
  quantiles.
* ``st.tl.cache.write_subset_h5ad`` takes ``obsp_keys``, and
  ``merge_h5ad_into_adata`` copies ``obsp``, so a cached CCI run keeps its
  neighbours.
* ``stlearn.tl.cci.neighbours.calc_neighbours`` returns that sparse matrix
  instead of a numba ``List``. ``neighbourhood_matrix`` and
  ``neighbourhoods_from_barcodes`` are removed.
* ``LRScorer`` takes a sparse matrix. ``get_scores``, ``adjacency`` and
  ``flatten_neighbours`` are removed from ``stlearn.tl.cci.scoring``.
* ``stlearn.tl.cci.neighbours.get_neighbourhoods`` and the edge helpers in
  ``stlearn.tl.cci.edges`` are removed.
* ``get_interaction_matrix`` and ``get_interaction_pvals`` take the graph's
  CSR arrays (``neigh_indptr``, ``neigh_indices``) instead of a numba ``List``.
* ``get_edges`` no longer needs to deduplicates in a quadratic loop.
* Removed ``UndefinedException`` from ``resistance_matrix``.

1.5.0 (2026-11-01)
------------------
This release contains changes in cell-cell interaction (CCI) mainly
improvements in runtime and memory usage and requires code to be updated.

* Memory usage is reduced through chunking (~256MB) in ``st.cci.run``
  scoring and permutation testing. On a 5,000,000 bin synthetic Visium HD
  sample with connectomeDB2025, n_pairs=1000, runs take ~24GB in ~10 minutes.
* SpatialData support added with a new ``st.io`` package. Supporting:
  ``atera``, ``xenium``, ``visium_hd`` and ``visium``. Start of
  spatial data native plotting with ``st.pl.sdata_plot`` and added
  AnnData's ``st.pp.highly_variable_genes``.
* New datasets ``st.datasets.visium_hd_sge`` and ``atera_sge``. All four
  dataset functions now return SpatialData objects; ``visium_sge`` returned
  AnnData and ``xenium_sge`` only downloaded the files.
* Upgraded ligand-receptor databases to connectomeDB2025 (human and mouse).
* Ligand-receptor pairs are an ``LRPairs`` object instead of an array of
  'L_R' strings. It tracks the species and where each pair came from.
  Pairs can be filtered and combined (``lrs.filter(...)``, ``lrs1 | lrs2``).

  * ``st.tl.cci.run`` takes only ``LRPairs``; an array of 'L_R' names raises
    ``TypeError``. Wrap strings with
    ``st.tl.cci.LRPairs.from_names(names, species="human")``.
  * ``st.tl.cci.lr_pairs(database, species)`` loads a database shipped with
    stLearn: ``"connectomeDB2025"`` (default; human or mouse) or
    ``"connectomeDB2020"`` (human).
  * ``st.tl.cci.read_lr_pairs(path, species=...)``
    reads your own csv/tsv of ligand and receptor columns.
  * ``st.tl.cci.load_lrs`` is deprecated and all APIs now use ``lr_pairs``,
    or ``lr_pairs(...).names()`` for the old array. Note that ``lr_pairs()``
    defaults to connectomeDB2025, where ``load_lrs()`` defaults to
    connectomeDB2020.
  * Previous databases names ``"connectomeDB2020_lit"`` and
    ``"connectomeDB2020_put"`` are removed. ``"connectomeDB2020"`` is the
    human only, literature pairs and the putative pairs are no longer shipped.
  * ``st.add.lr`` is removed; use ``read_lr_pairs`` or ``LRPairs.from_names``.
  * The LR table from ``run`` stores each pair's source in ``var["source"]``
    and the species in ``uns["lr_species"]``, which ``run_lr_go`` uses as its
    default species.
* Spatial trajectory and other APIs use cluster label strings
  (``cluster="6"``).
* AnnData backwards compatibility available through ``st.io.attach_spatial``
  for Visium/Visium HD and ``st.io.attach_spatial_sdata`` for Xenium/Atera.
* CCI results are now a LR table (AnnData) returned by ``st.tl.cci.run``.
  A spots x LR pairs with summaries in ``var`` and scores and p-values are
  in sparse ``layers``. Downstream functions and plots now take
  ``lr_results``.
* Support for microns through ``microns_per_pixel``,
  ``st.utils.microns_to_pixels``, ``pixels_to_microns`` and
  ``create_grid(bin_um=)``.
* SME computes weights only between nearby spots and data is kept sparse
  where possible.
* ``st.pp.extract_feature`` supports ConvNeXt and takes ``weights=``.
* Upgrade libraries: pillow>=13.0.0,<14.0, numba>=0.64.0,
  scikit-image>=0.25.0, torch>=2.5.0, matplotlib>=3.11,<4.0.

API and Bug Fixes:

* Fixed double application of scaling, adding instructions and warning for
  image alignment in ``st.io.create_stlearn`` Bug #303.
* A pair listed twice is now tested once - fixed Bug #305.
* Incorrect labelling in CCI Plot - rows labelled "Senders", columns labelled
  "Receivers" - should be the other way around Bugs #309, #363.
* SME failed with "Input X contains NaN" when a spot had no
  ``array_row``/``array_col`` - these are now left out of the spot spacing fit
  - fixed Bug #310.
* ``st.tl.cci.run(use_label=...)`` never applied cell heterogeneity weighting:
  it read the cell-type data from the wrong key, found nothing, and weighted
  every spot 1, so results were identical with and without ``use_label``. It
  now weights by heterogeneity from cell-type proportions in ``uns[use_label]``
  or per-cell labels in ``obs[use_label]``, and takes ``cell_prop_cutoff``.
  CCI scores and significance change for any run that used ``use_label``.
* ``st.tl.cci.run_cci``: p-values from permutation are now ``(k+1)/(N+1)``
  and can no longer be 0, cutoff is now ``p < p_cutoff``, and each
  permutation thread is seeded through ``random_state``.
* ``spot_mixtures`` matches cell types by exact name.
* When ``min_expr > 0`` ``n_spots`` in the LR summary counted spots with a
  score above ``min_expr``, while the ``min_spots`` filter counted spots
  with a score above 0, so the reported count was not the one the filter
  used. Both now use score > 0.
* ``st.tl.cci.grid`` dropped cells lying exactly on the inner edge of the
  last row or column of the grid, and on the outer edges of the corner
  square.
* ``st.pl.lr_plot`` statistic names were changed from ``p_val``, ``p_adj`` and
  ``-log10(p_adj)`` to ``p_vals``, ``p_adjs`` and ``neg_log10_p_adjs``. It
  always drew ``lr_sig_scores`` whatever ``use_label`` asked for, so a plot
  labelled ``p_adjs`` or ``neg_log10_p_adjs`` showed the masked score instead.
  ``st.pl.lr_plot`` and ``st.pl.ccinet_plot`` also draw on a ``fig`` or
  ``ax`` you passed in rather than opening a new figure, so they can go in
  subplots.
* Images in ``uns["spatial"]`` are now uint8 (0-255), not floats in [0, 1].
* ``st.pp.extract_feature`` tiles were fed as floats between 0-255 instead
  of 0-1. Tiles are now read as uint8, so ``X_tile_feature`` and
  ``X_morphology`` change.
* ``st.concatenate_spatial_adata``: the x and y offsets added to
  ``obsm["spatial"]`` for each tile were swapped, so spots on any tile off
  the diagonal landed on the wrong tile. Tiles also keep their aspect ratio
  instead of being stretched to ``fixed_size``, each sample is scaled by its
  own scale factor rather than the first sample's, and ``fixed_size`` is
  (height, width) like a NumPy shape.
* ``st.io.read_merfish`` works again with new notebook. Defaults
  ``spot_diameter_fullres`` to 10 (it was 50).
* ``st.io.read_10x`` reads 16-bit / signed PNGs and reports missing images
  clearly.
* ``st.add.labels`` / ``st.add.add_loupe_clusters`` match by barcode.
* ``st.add.positions`` reads Space Ranger 2.0+ files.
* ``st.add.parsing`` assigned spots the wrong coordinates (they were interleaved
  from the spots in ``obs``), so every plot and anything using
  ``obsm["spatial"]`` was wrong. Each spot now gets its own position.
* SME fits each axis separately when finding neighbours.
* Spatial trajectory: ``st.spatial.trajectory.pseudotime`` orients by median
  pseudotime (paths may differ), ``shortest_path_spatial_paga`` returns
  labels, plus fixes to ``local_level``, ``weight_optimizing_local``,
  ``de_transition_plot`` and the trajectory plots.
* ``st.pl.qc_plot`` no longer makes a dense copy.
* Readers were moved to ``st.io``: ``st.read_10x``, ``st.read_xenium``,
  ``st.read_old_st``, ``st.read_slide_seq``, ``st.read_merfish``,
  ``st.read_seq_fish`` and ``st.create_stlearn``.
* ``st.tl.cci.grid`` vectorised and no longer takes ``n_cpus``.
* ``obsm["spatial"]`` is in full-resolution pixels (was image pixels),
  matching the rest of stLearn (used by ``st.tl.cci.grid``).
* SME outputs are sparse when ``X`` is sparse; no weight matrices in ``uns``.
* Removed: ``extract_feature(cnn_base="xception")``,
  ``deconvolution_plot(data_alpha=)``, ``local_plot(cluster=)``,
  ``Spatial(basis=)``.
* New defaults: ``pl.trajectory.pseudotime_plot(node_size=60)`` (area);
  ``pl.trajectory.check_trajectory(size_spatial=7, img_key=None)``.
* ``st.spatial.trajectory.pseudotime`` no longer takes ``n_neighbors``,
  ``use_rep`` or ``run_knn`` - these must be run prior and will throw an
  exception if there's no neighbourhood graph.

1.4.1 (2026-06-16)
------------------
* Improved speed of random spot generation used in permutation testing.

API and Bug Fixes:
* Fixed up docstring and types for various CCI methods.
* Fixes for issues #350 and #300 with incorrect types.
* Upgrade libraries: pillow>=12.0.0,<13.0

1.4.0 (2026-05-05)
------------------
* Removed tensorflow and keras and replaced with torch and torchvision.
* The CCI implementation removed string comparisons for spot-pair edges and
  replaced it with one set of two packed 32 bit integers. Removed
  deduplication pass (m^2) with a hash lookup (O(n)).
* Upgrade libraries: numpy >=2.4.0, scanpy >= 1.12.0, scipy >= 1.17.0, added
  spatialdata, shapely, geopandas.

API and Bug Fixes:
* Added row and polygon annotations.
* Fixed exception handling to catch previous exceptions when rethrown.

1.3.0 (2026-02-24)
------------------
* Removed interactive stLearn/embedded web application.
* Removed louvain clustering - replaced with leiden.

API and Bug Fixes:
* Fix import on MutableVertexPartition to use leidenalg.VertexPartition.
* Switch default flavour in leiden to use igraph (and its required parameters).
* Renamed methods and classes to follow lowercase standard.

1.2.2 (2025-10-20)
------------------
* Added support for Python 3.11 and 3.12.
* Upgraded scanpy to 1.11 - clustering will be different.
* Added more CCI tests.

API and Bug Fixes:
* Fixed copy-paste error in ``leiden.py`` file.

1.1.5 (2025-09-17)
------------------
* Add Leiden clustering wrapper.
* Fix documentation, refactor code in spatial.SME.

1.1.1 (2025-07-07)
------------------
* Support Python 3.10.x
* Added quality checks black, ruff and mypy and fixed appropriate source code.
* Copy parameters now work with the same semantics as scanpy.
* Library upgrades for leidenalg, louvain, numba, numpy, scanpy, and tensorflow.
* datasets.xenium_sge - loads Xenium data (and caches it) similar to scanpy.visium_sge.

API and Bug Fixes:

* Xenium TIFF and cell positions are now aligned.
* Consistent with type annotations - mainly missing None annotations.
* pl.cluster_plot - Does not keep colours from previous runs when clustering.
* pl.trajectory.pseudotime_plot - Fix typing of cluster values in .uns["split_node"].
* Removed datasets.example_bcba - Replaced with wrapper for scanpy.visium_sge.
* Moved spatials directory to spatial, cleaned up pl and tl packages.

0.4.11 (2022-11-25)
-------------------

0.4.10 (2022-11-22)
-------------------

0.4.8 (2022-06-15)
------------------

0.4.7 (2022-03-28)
------------------

0.4.6 (2022-03-09)
------------------

0.4.5 (2022-03-02)
------------------

0.4.0 (2022-02-03)
------------------

0.3.2 (2021-03-29)
------------------

0.3.1 (2020-12-24)
------------------

0.2.7 (2020-09-12)
------------------

0.2.6 (2020-08-04)
------------------
