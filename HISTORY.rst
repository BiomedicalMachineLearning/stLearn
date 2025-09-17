=======
History
=======

1.1.3 (2025-09-17)
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
