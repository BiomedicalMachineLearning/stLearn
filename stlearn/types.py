from typing import Literal

_SIMILARITY_MATRIX = Literal["cosine", "euclidean", "pearson", "spearman"]
_METHOD = Literal["mean", "median", "sum"]
_QUALITY = Literal["fulres", "hires", "lowres"]
_BACKGROUND = Literal["black", "white"]

# The tissue image to draw behind a plot: "auto" chooses one, None draws none,
# and any other string is the name of an image, which must exist.
type ImageChoice = Literal["auto"] | str | None

__all__ = ["_BACKGROUND", "_METHOD", "_QUALITY", "_SIMILARITY_MATRIX", "ImageChoice"]
