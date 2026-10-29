"""Handling ligand-receptor pairs - a Database and LRPairs data structure for loading
and manipulating LR sources as well as functions to manipulate LRPairs.

Usage::

    # default database - connectomeDB2025, human
    connectome2025 = st.tl.cci.lr_pairs()
    mine = st.tl.cci.LRPairs.from_names(
        ["CCN2_ITGAV", "CCN2_ITGB3", "CXCL12_ITGA4", "CXCL12_ITGB1", "THBS1_CD47"],
        source="hand-picked",
    )
    extra = st.tl.cci.read_lr_pairs("pairs.tsv", sep="\t", species="human")
    lrs = connectome2025 | mine | extra
    lr_results = st.tl.cci.run(adata, lrs)
"""

from __future__ import annotations

import functools
import re
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.resources import files
from os import PathLike
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

DEFAULT_DATABASE = "connectomeDB2025"
SOURCE_SEP = " | "  # between the sources of a pair from more than one
_DB_DIR = files("stlearn.tl.cci") / "databases"


@dataclass(frozen=True)
class _Database:
    """An LR table shipped in stlearn/tl/cci/databases. `filename` may contain
    '{species}'; `species` is None when any species with a file is available."""

    filename: str
    sep: str
    ligand: int | str
    receptor: int | str
    species: tuple[str, ...]


_DATABASES = {
    "connectomeDB2025": _Database(
        "ConnectomeDB2025_{species}.csv",
        ",",
        "Ligand Symbols",
        "Receptor Symbols",
        species=("human", "mouse"),
    ),
    "connectomeDB2020": _Database(
        "connectomeDB2020.tsv",
        "\t",
        "Ligand gene symbol",
        "Receptor gene symbol",
        species=("human",),
    ),
}


class LRPairs:
    """An immutable set of ligand-receptor pairs of one species.

    Holds a table with 'ligand' and 'receptor' columns of gene symbols, one row per
    pair; a 'source' column saying where each pair came from; and any other columns
    of the source (e.g. connectomeDB2025's 'Evidence') to filter on. Methods return
    new LRPairs, so calls chain. Pairs combine only with pairs of the same species.

    Make one with lr_pairs() (databases shipped with stlearn), read_lr_pairs() (a
    csv/tsv file), LRPairs.from_frame() or LRPairs.from_names().
    """

    __slots__ = ("_frame", "_sources", "_species")

    def __init__(self, frame: pd.DataFrame, *, species: str, source: str = "DataFrame"):
        """`frame` needs 'ligand' and 'receptor' columns; rows without a 'source'
        get `source`. Symbols are reduced to the official one (dropping bracketed
        aliases), rows without a symbol on both sides are dropped, and a repeated
        pair becomes one row listing all its sources."""
        missing = {"ligand", "receptor"} - set(frame.columns)
        if missing:
            raise ValueError(
                f"LRPairs needs 'ligand' and 'receptor' columns, missing {missing}; "
                "use LRPairs.from_frame(df, ligand=..., receptor=..., species=...)",
            )
        if not species or not isinstance(species, str):
            raise ValueError(
                f"species must name a species, e.g. 'human'; got {species!r}",
            )
        if "source" not in frame.columns:
            frame = frame.assign(source=source)
        frame = frame.assign(
            ligand=_official_symbols(frame["ligand"]),
            receptor=_official_symbols(frame["receptor"]),
            source=frame["source"].astype("string").fillna(source),
        )
        frame = frame[_is_symbol(frame["ligand"]) & _is_symbol(frame["receptor"])]
        ligands, receptors = frame["ligand"], frame["receptor"]
        with_underscore = ligands.str.contains("_") | receptors.str.contains("_")
        if with_underscore.any():
            examples = frame.loc[with_underscore, ["ligand", "receptor"]].head(3)
            raise ValueError(
                f"Gene symbols in {source} contain '_', which separates ligand from "
                f"receptor in 'L_R' pair names; e.g. {examples.values.tolist()}",
            )
        frame = _merge_repeats(frame).reset_index(drop=True)
        self._set(frame, species.lower(), _labels(frame))

    def _set(self, frame: pd.DataFrame, species: str, sources: tuple[str, ...]):
        self._frame, self._species, self._sources = frame, species, sources

    def _derive(self, frame: pd.DataFrame, sources: tuple[str, ...]) -> LRPairs:
        """New LRPairs of this species from an already clean table; `sources` in the
        order they were added, keeping those that still have pairs."""
        present = set(_labels(frame))
        pairs = object.__new__(LRPairs)
        kept = tuple(label for label in sources if label in present)
        pairs._set(frame.reset_index(drop=True), self.species, kept)
        return pairs

    @property
    def species(self) -> str:
        """The species of the gene symbols, e.g. 'human'."""
        return self._species

    @property
    def sources(self) -> tuple[str, ...]:
        """Every source of these pairs, in the order they were added."""
        return self._sources

    # Constructors
    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        ligand: int | str = 0,
        receptor: int | str = 1,
        *,
        species: str,
        source: str = "DataFrame",
    ) -> LRPairs:
        """Pairs from two columns of a table, given by position or by name (matched
        ignoring case, spaces and underscores), labelled with `source`. Other
        columns are kept; a column already called 'source' is renamed
        'original_source'."""
        ligand_col, receptor_col = _find_column(df, ligand), _find_column(df, receptor)
        if ligand_col == receptor_col:
            raise ValueError(f"Ligand and receptor are both column {ligand_col!r}")
        frame = df.drop(columns=[ligand_col, receptor_col])
        frame = frame.rename(columns={"source": "original_source"})
        frame.insert(0, "ligand", df[ligand_col])
        frame.insert(1, "receptor", df[receptor_col])
        frame.insert(2, "source", source)
        pairs = cls(frame, species=species, source=source)
        if len(pairs) == 0:
            raise ValueError(
                f"No ligand-receptor pairs found in {source} (columns "
                f"{ligand_col!r} and {receptor_col!r}).",
            )
        return pairs

    @classmethod
    def from_names(
        cls,
        names: Iterable[str],
        *,
        species: str,
        source: str = "names",
    ) -> LRPairs:
        """Pairs from 'LIGAND_RECEPTOR' names, the format stlearn's results use."""
        return cls(_split_names(names), species=species, source=source)

    # Transformations; each returns a new LRPairs of the same species
    def filter(
        self,
        keep: Callable[[pd.DataFrame], npt.ArrayLike] | npt.ArrayLike,
    ) -> LRPairs:
        """The pairs whose rows `keep` marks True. `keep` is a boolean mask or a
        function of the table returning one, e.g. lambda d: d["Evidence"] == "Direct"
        or lambda d: d["source"].str.contains("connectomeDB2025").
        """
        mask = keep(self.to_frame()) if callable(keep) else keep
        # Positional, with missing values (e.g. a comparison with NaN) as False.
        flags = pd.Series(np.asarray(mask, dtype=object), dtype="boolean")
        kept = self._frame[flags.fillna(False).to_numpy(dtype=bool)]
        return self._derive(kept, self.sources)

    def union(self, *others: LRPairs) -> LRPairs:
        """Pairs in any of these, which must all be of this species. A pair in more
        than one lists every source; its other columns take the first value given.
        """
        others_species = {other.species for other in others} - {self.species}
        if others_species:
            described = ", ".join(
                f"{pairs.species} ({', '.join(pairs.sources)})"
                for pairs in (self, *others)
            )
            raise ValueError(
                f"Can't combine LR pairs of different species: {described}. "
                "Convert the genes to one species' orthologs first.",
            )
        frames = [self._frame] + [other._frame for other in others]
        merged = _merge_repeats(pd.concat(frames, ignore_index=True))
        sources = self.sources + tuple(s for other in others for s in other.sources)
        return self._derive(merged, tuple(dict.fromkeys(sources)))

    def __or__(self, other: LRPairs) -> LRPairs:
        if not isinstance(other, LRPairs):
            return NotImplemented
        return self.union(other)

    # Outputs
    def names(self) -> npt.NDArray[np.str_]:
        """'LIGAND_RECEPTOR' names, sorted."""
        return np.array(sorted(_names(self._frame)), dtype=np.str_)

    def genes(self) -> npt.NDArray[np.str_]:
        """Every ligand and receptor gene, sorted; e.g. a GO background."""
        genes = pd.concat([self._frame["ligand"], self._frame["receptor"]])
        return np.array(sorted(set(genes)), dtype=np.str_)

    def source_of(self, names: Iterable[str]) -> npt.NDArray[np.object_]:
        """The source of each 'LIGAND_RECEPTOR' name; None for pairs not in here."""
        by_name = dict(zip(_names(self._frame), self._frame["source"], strict=True))
        return np.array([by_name.get(str(name)) for name in names], dtype=object)

    def to_frame(self) -> pd.DataFrame:
        """A copy of the table: 'ligand', 'receptor', 'source', and the source's
        other columns."""
        return self._frame.copy()

    def __len__(self) -> int:
        return len(self._frame)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str) or name.count("_") != 1:
            return False
        ligand, receptor = name.split("_")
        frame = self._frame
        return bool(
            ((frame["ligand"] == ligand) & (frame["receptor"] == receptor)).any(),
        )

    def __repr__(self) -> str:
        sources = ", ".join(self.sources) or "nowhere"
        return f"LRPairs({len(self)} {self.species} pairs from {sources})"


def load_lrs(
    names: str | list[str] | None = None,
    species: str = "human",
) -> npt.NDArray[np.str_]:
    """Deprecated: use lr_pairs(), which returns an LRPairs, and lr_pairs(...).names()
    for this function's array.

    Loads inputted LR database, & concatenates into consistent database set of
    pairs without duplicates. If None loads 'connectomeDB2020'.

    Parameters
    ----------
    names: list
        Databases to load, options: 'connectomeDB2020' and 'connectomeDB2025'.
        If more than one specified, loads all & removes duplicates.
    species: str
        Format of the LR genes, either 'human' or 'mouse'.
    Returns
    -------
    lrs: np.ndarray
       lr pairs from the database in format ['L1_R1', 'LN_RN']
    """
    warnings.warn(
        "load_lrs is deprecated; use st.tl.cci.lr_pairs(database, species), which "
        "returns LRPairs that run() and run_lr_go(bg_genes=lrs.genes()) take.",
        FutureWarning,
        stacklevel=2,
    )
    if names is None:
        names = ["connectomeDB2020"]
    if isinstance(names, str):
        names = [names]

    pairs = lr_pairs(names[0], species=species).union(
        *(lr_pairs(name, species=species) for name in names[1:]),
    )
    return pairs.names()


def lr_pairs(database: str = DEFAULT_DATABASE, species: str = "human") -> LRPairs:
    """Ligand-receptor pairs from a database shipped with stlearn.

    Parameters
    ----------
    database: str
        'connectomeDB2025' (default; human or mouse; Liu et al. 2025, NAR) or
        'connectomeDB2020' (human; Hou et al. 2020, Nat Commun).
    species: str
        Species of the gene symbols. Each species has its own table, so mouse pairs
        use mouse symbols and include mouse-only pairs.
    Returns
    -------
    LRPairs
        With the database's name as each pair's source.
    """
    spec = _DATABASES.get(database)
    if spec is None:
        raise ValueError(
            f"Unknown LR database {database!r}; options are {list(_DATABASES)}.",
        )
    species = species.lower()
    if species not in spec.species:
        raise ValueError(f"{database} only has {' and '.join(spec.species)} LRs.")
    return _read_packaged(database, species)


def read_lr_pairs(
    path: str | PathLike[str],
    ligand: int | str = 0,
    receptor: int | str = 1,
    *,
    species: str,
    sep: str | None = None,
    source: str | None = None,
) -> LRPairs:
    """Ligand-receptor pairs from a csv or tsv file.

    Parameters
    ----------
    path: str or Path
        The file; may be compressed (.gz, .bz2, .zip, .xz).
    ligand, receptor: int or str
        The columns holding the ligand and receptor symbols, by position or by name
        (matched ignoring case, spaces and underscores). A symbol followed by
        bracketed aliases, 'CCN2 (CTGF, HCS24)', is read as 'CCN2'.
    species: str
        The species of the gene symbols, e.g. 'human' or 'mouse'.
    sep: str
        The delimiter. If None, ',' for .csv, tab for .tsv/.txt/.tab, and otherwise
        detected from the file.
    source: str
        Each pair's source; if None, the file's name.
    Returns
    -------
    LRPairs
    """
    sep = sep or _sep_for(path)
    if sep is None:
        df = pd.read_csv(path, sep=None, engine="python")
    else:
        df = pd.read_csv(path, sep=sep)
    return LRPairs.from_frame(
        df,
        ligand,
        receptor,
        species=species,
        source=source or Path(path).name,
    )


def _as_lr_pairs(lrs: LRPairs | Iterable[str]) -> LRPairs:
    """Converts everything to LRPairs - assumes Iterable[str] are human and
    hand-picked."""
    if isinstance(lrs, LRPairs):
        return lrs
    if isinstance(lrs, str | PathLike | pd.DataFrame):
        raise TypeError(
            f"Expected LRPairs or 'L_R' names, got {type(lrs).__name__}; use "
            "lr_pairs(database), read_lr_pairs(path) or LRPairs.from_frame(df).",
        )
    lrs = LRPairs.from_names(
        lrs,
        species="unknown",
        source="hand-picked",
    )
    return lrs


# Helpers
_COMPRESSION = {".gz", ".bz2", ".zip", ".xz", ".zst"}
_SEPARATORS = {".csv": ",", ".tsv": "\t", ".txt": "\t", ".tab": "\t"}


@functools.cache
def _read_packaged(database: str, species: str) -> LRPairs:
    spec = _DATABASES[database]
    with (_DB_DIR / spec.filename.format(species=species)).open("rb") as fh:
        df = pd.read_csv(fh, sep=spec.sep)
    return LRPairs.from_frame(
        df,
        spec.ligand,
        spec.receptor,
        species=species,
        source=database,
    )


def _sep_for(path: str | PathLike[str]) -> str | None:
    suffixes = [suffix.lower() for suffix in Path(path).suffixes]
    if suffixes and suffixes[-1] in _COMPRESSION:
        suffixes = suffixes[:-1]
    return _SEPARATORS.get(suffixes[-1]) if suffixes else None


def _find_column(df: pd.DataFrame, column: int | str) -> str:
    """A column's name, given its position or its name ignoring case, spaces and
    underscores (connectomeDB2025 has 'Ligand Symbols' in the current release and
    'Ligand_Symbols' in CDB2025v1)."""
    if isinstance(column, int):
        if column >= df.shape[1]:
            raise ValueError(
                f"LR table has {df.shape[1]} column(s); needs ligand and receptor",
            )
        return df.columns[column]

    def key(name: object) -> str:
        return str(name).lower().replace(" ", "").replace("_", "")

    for name in df.columns:
        if key(name) == key(column):
            return name
    raise ValueError(f"No {column!r} column in LR table; has {list(df.columns)}")


def _official_symbols(cells: pd.Series) -> pd.Series:
    """The official symbol from each cell; connectomeDB2025 follows it with old names
    and aliases in brackets, 'CCN2 (CTGF, HCS24)'."""
    return (
        cells.astype("string")
        .str.replace(r"<[^>]*>", "", regex=True)  # in case of HTML-table exports
        .str.split(" (", n=1, regex=False)
        .str[0]
        .str.strip()
    )


def _is_symbol(cells: pd.Series) -> pd.Series:
    """One symbol: no spaces, so 'no human ortholog' rows are dropped."""
    return cells.str.fullmatch(r"\S+").fillna(False).astype(bool)


def _labels(frame: pd.DataFrame) -> tuple[str, ...]:
    """The sources named in a table's 'source' column, in order of appearance."""
    labels = frame["source"].str.split(SOURCE_SEP, regex=False).explode()
    return tuple(dict.fromkeys(labels.dropna()))


def _names(frame: pd.DataFrame) -> pd.Series:
    return frame["ligand"] + "_" + frame["receptor"]


def _split_names(names: Iterable[str]) -> pd.DataFrame:
    """'LIGAND_RECEPTOR' names as a ligand/receptor table, checking their format."""
    names = [str(name) for name in names]
    bad = [name for name in names if not re.fullmatch(r"[^_\s]+_[^_\s]+", name)]
    if bad:
        raise ValueError(f"LR names must look like 'LIGAND_RECEPTOR'; got {bad[:3]}")
    split = [name.split("_") for name in names]
    return pd.DataFrame(split, columns=["ligand", "receptor"], dtype="string")


def _merge_repeats(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per pair: a repeated pair lists all its sources, and takes the first
    non-missing value of each other column."""
    keys = ["ligand", "receptor"]
    if not frame.duplicated(keys).any():
        return frame
    grouped = frame.groupby(keys, sort=False)
    merged = grouped.first()
    merged["source"] = grouped["source"].agg(
        lambda sources: SOURCE_SEP.join(
            dict.fromkeys(
                label
                for joined in sources.dropna()
                for label in joined.split(SOURCE_SEP)
            ),
        ),
    )
    return merged.reset_index()
