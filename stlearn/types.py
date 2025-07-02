from typing import Literal

_SIMILARITY_MATRIX = Literal["cosine", "euclidean", "pearson", "spearman"]
_METHOD = Literal["mean", "median", "sum"]

__all__ = ["_SIMILARITY_MATRIX", "_METHOD"]
