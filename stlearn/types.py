from typing import Literal

_SIMILARITY_MATRIX = Literal["cosine", "euclidean", "pearson", "spearman"]
_METHOD = Literal["mean", "median", "sum"]
_QUALITY = Literal["fulres", "hires", "lowres"]
_BACKGROUND = Literal["black", "white"]

__all__ = [
    "_SIMILARITY_MATRIX",
    "_METHOD",
    "_QUALITY",
    "_BACKGROUND"
]
