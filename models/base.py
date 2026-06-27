"""Abstract model interface shared by the baseline and sequence twins.

A :class:`DecisionModel` predicts the next decision for one domain and exposes a
calibrated probability distribution over that domain's option set, so confidence and
Brier-score calibration are well defined. Concrete models persist to / load from an
artifact path with prediction-equivalent round-trips.

This module also provides the small, model-family-agnostic helpers that the baseline
(task 6.2) and sequence (task 8.1) twins share:

* :class:`LabelSpace` -- the ordered per-domain option set plus a stable
  label<->index mapping (the model label space),
* :func:`label_space` -- build a :class:`LabelSpace` directly from a domain using the
  canonical option ordering in :data:`data.schema.DOMAIN_OPTIONS`,
* :func:`is_valid_distribution` / :func:`validate_distribution` -- check that a
  ``dict[str, float]`` or array is a valid probability distribution (non-negative and
  sums to ~1, supporting Requirement 5.3),
* :func:`build_class_probs` -- assemble a valid, full-coverage ``class_probs`` dict over
  a domain's option set (used by every model's ``predict_proba``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from data.schema import Domain, options

#: Absolute tolerance for individual probabilities being treated as non-negative.
_NEG_TOL: float = 1e-6

#: Absolute tolerance for the "sums to ~1" check on a probability distribution.
_SUM_TOL: float = 1e-3


# ---------------------------------------------------------------------------
# Per-domain label space
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelSpace:
    """The ordered option set for one domain plus a stable label<->index mapping.

    The ordering is the canonical per-domain option ordering from
    :data:`data.schema.DOMAIN_OPTIONS`, so the index assigned to each label is stable
    across fit / predict / save / load and shared by both model families.

    Attributes:
        labels: The ordered tuple of valid decision options for the domain.
    """

    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"label space contains duplicate labels: {self.labels}")
        if not self.labels:
            raise ValueError("label space must contain at least one label")

    # -- size / membership ------------------------------------------------
    def __len__(self) -> int:
        return len(self.labels)

    def __iter__(self):
        return iter(self.labels)

    def __contains__(self, label: object) -> bool:
        return label in self.labels

    @property
    def size(self) -> int:
        """Number of options in this label space."""
        return len(self.labels)

    # -- mappings ---------------------------------------------------------
    @property
    def label_to_index(self) -> dict[str, int]:
        """Mapping from each label to its stable integer index."""
        return {label: i for i, label in enumerate(self.labels)}

    @property
    def index_to_label(self) -> dict[int, str]:
        """Mapping from each integer index back to its label."""
        return dict(enumerate(self.labels))

    def encode(self, label: str) -> int:
        """Return the stable index for ``label``.

        Raises:
            KeyError: if ``label`` is not part of this label space.
        """
        try:
            return self.label_to_index[label]
        except KeyError as exc:
            raise KeyError(
                f"label {label!r} is not in label space {self.labels}"
            ) from exc

    def decode(self, index: int) -> str:
        """Return the label for a stable ``index``.

        Raises:
            IndexError: if ``index`` is out of range for this label space.
        """
        if not 0 <= index < len(self.labels):
            raise IndexError(
                f"index {index} out of range for label space of size {len(self.labels)}"
            )
        return self.labels[index]


def label_space(domain: str | Domain) -> LabelSpace:
    """Return the :class:`LabelSpace` for ``domain`` using the canonical ordering.

    Raises:
        ValueError: if ``domain`` is not a supported domain.
    """
    return LabelSpace(tuple(options(domain)))


# ---------------------------------------------------------------------------
# Probability-distribution helpers
# ---------------------------------------------------------------------------


def _as_values(probs: Mapping[str, float] | Sequence[float] | np.ndarray) -> list[float]:
    """Coerce a probability dict / sequence / array into a flat list of floats."""
    if isinstance(probs, Mapping):
        raw: Iterable[float] = probs.values()
    elif isinstance(probs, np.ndarray):
        raw = probs.ravel().tolist()
    else:
        raw = probs
    return [float(v) for v in raw]


def is_valid_distribution(
    probs: Mapping[str, float] | Sequence[float] | np.ndarray,
    tol: float = _SUM_TOL,
) -> bool:
    """Return ``True`` if ``probs`` is a valid probability distribution.

    A distribution is valid when every entry is non-negative (within
    :data:`_NEG_TOL`) and the entries sum to approximately ``1.0`` (within ``tol``).
    Accepts a ``dict[str, float]``, any sequence of floats, or a numpy array.
    """
    values = _as_values(probs)
    if not values:
        return False
    if any(v < -_NEG_TOL for v in values):
        return False
    return abs(sum(values) - 1.0) <= tol


def validate_distribution(
    probs: Mapping[str, float] | Sequence[float] | np.ndarray,
    tol: float = _SUM_TOL,
) -> None:
    """Raise ``ValueError`` if ``probs`` is not a valid probability distribution.

    See :func:`is_valid_distribution` for the validity rules. Kept as a raising
    counterpart so callers in ``predict_proba`` can assert validity inline.
    """
    values = _as_values(probs)
    if not values:
        raise ValueError("class_probs is empty")
    if any(v < -_NEG_TOL for v in values):
        raise ValueError(f"class_probs contains negative values: {probs}")
    total = float(sum(values))
    if abs(total - 1.0) > tol:
        raise ValueError(f"class_probs must sum to ~1.0, got {total}: {probs}")


def build_class_probs(
    space: LabelSpace | Sequence[str],
    raw: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Build a valid ``class_probs`` dict covering every option in ``space``.

    Ensures full option coverage (missing options default to ``0.0``), drops any keys
    that are not part of the label space, and normalizes so the result sums to ``1.0``.
    When ``raw`` is ``None`` or sums to ``0`` a uniform distribution over the options is
    returned, so the output is always a valid distribution per
    :func:`is_valid_distribution`.

    Args:
        space: A :class:`LabelSpace` or an ordered sequence of option labels.
        raw: Optional (possibly partial / unnormalized) per-label scores.

    Returns:
        A normalized ``dict[str, float]`` keyed by every option in ``space``.
    """
    labels = list(space.labels) if isinstance(space, LabelSpace) else list(space)
    if not labels:
        raise ValueError("cannot build class_probs over an empty option set")

    probs = {label: 0.0 for label in labels}
    if raw:
        for label, value in raw.items():
            if label in probs:
                probs[label] += float(value)

    total = sum(probs.values())
    if total <= 0.0:
        uniform = 1.0 / len(labels)
        return {label: uniform for label in labels}
    return {label: value / total for label, value in probs.items()}


# ---------------------------------------------------------------------------
# Abstract model
# ---------------------------------------------------------------------------


class DecisionModel(ABC):
    """Abstract behavioral-twin model for a single domain.

    Concrete subclasses (the baseline and sequence twins) consume the engineered
    features produced by :class:`features.feature_pipeline.FeaturePipeline` -- a flat
    feature matrix ``X`` and a last-K sequence tensor ``seq`` -- and predict the next
    decision for one domain over a fixed :attr:`options` set.
    """

    name: str = "abstract"

    def __init__(self, options: list[str]) -> None:
        self.options = list(options)
        #: Stable label space derived from the option ordering.
        self.label_space = LabelSpace(tuple(self.options))

    @abstractmethod
    def fit(self, X: np.ndarray, seq: np.ndarray, y: list[str]) -> "DecisionModel":
        """Fit the model on flat features ``X``, sequence tensor ``seq``, and labels."""

    @abstractmethod
    def predict_proba(self, x: np.ndarray, seq: np.ndarray) -> dict[str, float]:
        """Return a probability distribution over the domain options for one sample."""

    def predict(self, x: np.ndarray, seq: np.ndarray) -> str:
        """Return the argmax option for one sample."""
        probs = self.predict_proba(x, seq)
        return max(probs, key=probs.get)

    def confidence(self, x: np.ndarray, seq: np.ndarray) -> float:
        """Return the maximum class probability for one sample."""
        return max(self.predict_proba(x, seq).values())

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the fitted model to ``path``."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "DecisionModel":
        """Load a model previously persisted with :meth:`save`."""
