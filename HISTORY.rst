=======
History
=======

1.5.0 (2026-10-29)
------------------
* Memory usage is reduced through chunking in CCI scoring and permutation
  testing ~5,000,000 bins @ 20GB in ~20 minutes.
* SpatialData support added with a new ``st.io`` package. Supporting:
  ``atera``, ``xenium``, ``visium_hd`` and ``visium``. Start of
  spatial data native plotting with ``st.pl.sdata_plot`` and
  ``st.pp.highly_variable_genes``.
* Spatial trajectory and other APIs use cluster labels strings
  (``cluster="6"``).
* Upgraded ligand-receptor databases to connectomeDB2025 (human and mouse).
* Loading via ``load_lrs`` is deprecated for ``lr_pairs`` with
  ``connectomeDB2020_put`` removed. CCI and GO results change with the
  new databases.
* New datasets available to download via ``visium_hd_sge`` and ``atera_sge``
  and return SpatialData objects.
* AnnData backwards compatibility available through ``st.io.attach_spatial``.
* CCI results are now a LR table (AnnData) returned by ``st.tl.cci.run``.
  A spots x LR pairs with summaries, scores and p-values in ``var```.
  Downstream functions and plots now take ``lr_results``.
* Support for microns through ``microns_per_pixel``,
  ``st.utils.microns_to_pixels``, ``pixels_to_microns`` and
  ``create_grid(bin_um=)``.
* SME computes weights only between nearby spots and data is kept sparse
  where possible.
* ``extract_feature`` supports ConvNeXt and takes ``weights=``.

API and Bug Fixes:

* ``st.tl.cci.run(use_label=...)`` never applied cell heterogeneity weighting:
  it read the cell-type data from the wrong key, found nothing, and weighted
  every spot 1, so results were identical with and without ``use_label``. It
  now weights by heterogeneity from cell-type proportions in ``uns[use_label]``
  or per-cell labels in ``obs[use_label]``, and takes ``cell_prop_cutoff``.
  CCI scores and significance change for any run that used ``use_label``.
* ``run_cci`` changes: p-values calculated via permutation calculation
  via  are ``(k+1)/(N+1)`` and can nolonger be 0, cutoff is now
  ``p < p_cutoff``, and permutations are seeded through ``random_state``.
* ``spot_mixtures`` matches cell types by exact name.
* ``n_spots`` in the LR summary counted spots with a score above ``min_expr``,
  while the ``min_spots`` filter counted spots with a score above 0, so the
  reported count was not the one the filter used. Both now use score > 0;
  ``n_spots`` goes up for weakly scoring pairs.
* ``st.tl.cci.grid`` dropped cells that fell on the last row or column of the
  grid or exactly on a square boundary. Every cell now lands in exactly one
  square, so cell counts, summed expression and label proportions change,
  mostly in edge squares.
* ``lr_plot`` always drew ``lr_sig_scores`` whatever ``use_label`` asked for,
  so a plot labelled ``p_adjs`` or ``neg_log10_p_adjs`` showed the masked
  score instead. It now draws the statistic requested. ``lr_plot`` and
  ``ccinet_plot`` also draw on a ``fig`` or ``ax`` you pass in rather than
  opening a new figure, so they can go in subplots.
* ``lr_plot`` / ``ccinet_plot`` use the ``fig`` / ``ax`` passed in rather
  than creating their own.
* Images in ``uns["spatial"]`` are now uint8 (0-255), not floats in [0, 1].
* ``extract_feature`` tiles were fed as floats between 0-255 instead of 0-1.
  Tiles are now read as uint8, so ``X_tile_feature`` and ``X_morphology``
  change.
* ``concatenate_spatial_adata``: the x and y offsets added to ``obsm["spatial"]``
  for each tile were swapped, so spots landed on the wrong tile unless the tile
  grid was symmetric. Tiles also keep their aspect ratio instead of being
  stretched to ``fixed_size``, each sample is scaled by its own scale factor
  rather than the first sample's, and ``fixed_size`` is (height, width) like a
  NumPy shape.
* ``read_merfish`` works again with new notebook.
* ``read_10x`` reads 16-bit / signed PNGs and reports missing images clearly.
* ``add.labels`` / ``add_loupe_clusters`` match by barcode
* ``add.positions`` reads Space Ranger 2.0+ files
* ``add.parsing`` assigned spots the wrong coordinates (they were written in a
  different order from the spots in ``obs``), so every plot and anything using
  ``obsm["spatial"]`` (neighbours, SME, CCI) was wrong. Each spot now gets its
  own position.
* SME fits each axis separately when finding neighbours.
* Spatial trajectory: ``pseudotime`` orients by median pseudotime (paths may
  differ), ``shortest_path_spatial_paga`` returns labels, plus fixes to
  ``local_level``, ``weight_optimizing_local``, ``de_transition_plot`` and the
  trajectory plots.
* Upgraded to current matplotlib (3.11.0).
* ``qc_plot`` no longer makes a dense copy.
* ``stlearn.wrapper`` removed.
* ``cci.grid`` vectorised and nolonger takes ``n_cpus``.
* ``obsm["spatial"]`` is in full-resolution pixels (was image pixels), matching
  the rest of stLearn.
* SME outputs are sparse when ``X`` is sparse; no weight matrices in ``uns``.
* Removed: ``extract_feature(cnn_base="xception")``,
  ``deconvolution_plot(data_alpha=)``, ``local_plot(cluster=)``,
  ``Spatial(basis=)``.
* New defaults: trajectory ``node_size=60`` (area);
  ``check_trajectory(size_spatial=7, img_key=None)``.

1.4.1 (2026-06-16)
------------------
* Improved speed of random spot generation used in permutation testing.

API and Bug Fixes:
* Fixed up docstring and types for various CCI methods.
* Fixes for issues #350 and #300 with incorrect types.
* Upgrade libraries: pillow>=12.0.0,<13.0

1.4.0 (2026-05-01)
------------------
* Removed tensorflow and keras and replaced with torch and torchvision.
* The CCI implementation removed string comparisons for LR pairs lists and replaced it with one set of two
  packed 32 bit integers. Removed deduplication pass (m^2) with a hash lookup (O(n)).
* Upgrade libraries: numpy >=2.4.0, scanpy >= 1.12.0, scipy >= 1.17.0, added spatialdata, shapely, geopandas.

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
* Fixed copy-paste error in louvain.py file.

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
------------------

0.4.10 (2022-11-22)
------------------

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
