from typing import Literal

_SIMILARITY_MATRIX = Literal["cosine", "euclidean", "pearson", "spearman"]
_METHOD = Literal["mean", "median", "sum"]
_QUALITY = Literal["fulres", "hires", "lowres"]
_BACKGROUND = Literal["black", "white"]

__all__ = ["_BACKGROUND", "_METHOD", "_QUALITY", "_SIMILARITY_MATRIX"]
