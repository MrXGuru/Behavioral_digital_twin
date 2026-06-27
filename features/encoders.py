"""Deterministic categorical encoders for the feature pipeline.

This module implements small, dependency-light one-hot encoders with an explicit
``<UNK>`` bucket for values unseen during :meth:`OneHotEncoder.fit`. Keeping the encoders
deterministic and self-contained guarantees the feature pipeline produces fixed,
schema-versioned dimensions (Requirement 4.5) and never raises on unseen categories at
prediction time (Requirement 4.6).

Two public encoders are provided:

* :class:`OneHotEncoder` -- encodes a single categorical column.
* :class:`ContextEncoder` -- composes several :class:`OneHotEncoder` instances to encode a
  group of categorical context columns (e.g. ``location`` + ``weather``) into one
  fixed-width vector.

The vocabulary learned during ``fit`` is sorted so the column layout is byte-for-byte
reproducible across runs, and a trailing ``<UNK>`` slot is always reserved so that the
output width is stable regardless of which values appear at transform time.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from data.schema import SCHEMA_VERSION, UNK


def _field(row: Any, key: str) -> Any:
    """Read ``key`` from ``row`` whether it is a mapping or an attribute holder.

    Supports plain dicts (``row[key]``) and objects such as :class:`DecisionRecord`
    (``row.key``) so the encoders work uniformly across the training and prediction paths.
    """
    if isinstance(row, Mapping):
        return row[key]
    return getattr(row, key)


class OneHotEncoder:
    """One-hot encode a single categorical column with an explicit ``<UNK>`` bucket.

    The vocabulary is learned during :meth:`fit` (sorted for determinism) and a trailing
    ``<UNK>`` slot is always reserved so that values unseen during fit map to it instead
    of raising. Output width is therefore ``len(vocabulary) + 1`` and is stable across
    calls once fit (Requirements 4.5, 4.6).
    """

    def __init__(self) -> None:
        self.vocabulary: list[str] = []
        self._index: dict[str, int] = {}
        self._fitted = False
        #: Schema version this encoder's layout is pinned to.
        self.schema_version: str = SCHEMA_VERSION

    # -- introspection ---------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._fitted

    @property
    def width(self) -> int:
        """Number of output columns (vocabulary size plus the ``<UNK>`` slot)."""
        return len(self.vocabulary) + 1

    @property
    def output_dim(self) -> int:
        """Fixed output dimension; alias of :attr:`width` (Requirement 4.5)."""
        return self.width

    @property
    def unk_index(self) -> int:
        """Index of the reserved ``<UNK>`` slot (always the trailing column)."""
        return len(self.vocabulary)

    # -- fit / transform -------------------------------------------------
    def fit(self, values: Iterable[Any]) -> "OneHotEncoder":
        """Learn the sorted, de-duplicated vocabulary from ``values``."""
        uniq = sorted({str(v) for v in values})
        self.vocabulary = uniq
        self._index = {v: i for i, v in enumerate(uniq)}
        self._fitted = True
        return self

    def index_of(self, value: Any) -> int:
        """Return the column index for ``value`` (the ``<UNK>`` index if unseen)."""
        if not self._fitted:
            raise RuntimeError("OneHotEncoder.index_of called before fit")
        return self._index.get(str(value), self.unk_index)

    def transform_one(self, value: Any) -> list[float]:
        """Return the one-hot vector (as a list) for ``value``.

        Unseen values map to the ``<UNK>`` slot rather than raising (Requirement 4.6).
        """
        if not self._fitted:
            raise RuntimeError("OneHotEncoder.transform_one called before fit")
        vec = [0.0] * self.width
        vec[self.index_of(value)] = 1.0
        return vec

    def transform_vector(self, value: Any) -> np.ndarray:
        """Return the one-hot encoding of a single ``value`` as a float32 array."""
        return np.asarray(self.transform_one(value), dtype=np.float32)

    def transform(self, values: Sequence[Any]) -> np.ndarray:
        """Return an ``(n, output_dim)`` float32 matrix for a sequence of values."""
        if not self._fitted:
            raise RuntimeError("OneHotEncoder.transform called before fit")
        if len(values) == 0:
            return np.zeros((0, self.width), dtype=np.float32)
        return np.asarray([self.transform_one(v) for v in values], dtype=np.float32)

    def feature_names(self, prefix: str) -> list[str]:
        """Return human-readable column names for this encoder."""
        return [f"{prefix}={v}" for v in self.vocabulary] + [f"{prefix}={UNK}"]


class ContextEncoder:
    """Encode a group of categorical context columns into one fixed-width vector.

    Composes one :class:`OneHotEncoder` per column (in a stable, caller-specified order)
    and concatenates their outputs. The total :attr:`output_dim` is the sum of the
    per-column widths and is fixed once :meth:`fit` has run, regardless of which values
    appear at transform time (Requirement 4.5). Unseen values in any column fall through
    to that column's ``<UNK>`` slot without raising (Requirement 4.6).
    """

    #: Default categorical context columns used as model context.
    DEFAULT_COLUMNS: tuple[str, ...] = ("location", "weather")

    def __init__(self, columns: Sequence[str] | None = None) -> None:
        self.columns: list[str] = list(
            self.DEFAULT_COLUMNS if columns is None else columns
        )
        self._encoders: dict[str, OneHotEncoder] = {
            c: OneHotEncoder() for c in self.columns
        }
        self._fitted = False
        #: Schema version this encoder's layout is pinned to.
        self.schema_version: str = SCHEMA_VERSION

    # -- introspection ---------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._fitted

    @property
    def output_dim(self) -> int:
        """Fixed total output dimension across all columns (Requirement 4.5)."""
        return sum(enc.width for enc in self._encoders.values())

    def encoder(self, column: str) -> OneHotEncoder:
        """Return the underlying :class:`OneHotEncoder` for ``column``."""
        return self._encoders[column]

    def column_widths(self) -> dict[str, int]:
        """Return the fixed per-column output widths."""
        return {c: self._encoders[c].width for c in self.columns}

    # -- fit / transform -------------------------------------------------
    def fit(self, rows: Iterable[Any]) -> "ContextEncoder":
        """Learn each column's vocabulary from an iterable of rows.

        Each row may be a mapping or an attribute holder (e.g. ``DecisionRecord``).
        """
        materialized = list(rows)
        for column in self.columns:
            values = [_field(row, column) for row in materialized]
            self._encoders[column].fit(values)
        self._fitted = True
        return self

    def transform_one(self, row: Any) -> np.ndarray:
        """Return the concatenated one-hot context vector for a single ``row``."""
        if not self._fitted:
            raise RuntimeError("ContextEncoder.transform_one called before fit")
        parts = [
            self._encoders[column].transform_one(_field(row, column))
            for column in self.columns
        ]
        flat = [v for part in parts for v in part]
        return np.asarray(flat, dtype=np.float32)

    def transform(self, rows: Sequence[Any]) -> np.ndarray:
        """Return an ``(n, output_dim)`` float32 matrix for a sequence of rows."""
        if not self._fitted:
            raise RuntimeError("ContextEncoder.transform called before fit")
        if len(rows) == 0:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        return np.asarray([self.transform_one(r) for r in rows], dtype=np.float32)

    def feature_names(self) -> list[str]:
        """Return human-readable column names for the concatenated vector."""
        names: list[str] = []
        for column in self.columns:
            names.extend(self._encoders[column].feature_names(column))
        return names
