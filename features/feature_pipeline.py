"""Feature pipeline orchestration for the Behavioral Digital Twin.

Combines temporal, context, history, and per-user profile features into model-ready
matrices for a single decision domain. The pipeline is deterministic and leakage-free:
for the record at index ``i`` the history and rolling-frequency components reflect only
decisions strictly earlier than ``i``.

Each flat feature vector concatenates, in a fixed schema-versioned order:

``[ temporal | rolling-frequency | context | last-K history one-hot | user embedding ]``

Two output representations are produced from one pass:

* ``X`` -- a flat 2-D float matrix for the gradient-boosting / logistic baseline,
* ``seq`` -- a 3-D ``(n, K, step_dim)`` tensor for the sequence model,

so both model families consume consistently engineered features. The flat width and the
sequence step width are fixed once :meth:`FeaturePipeline.fit` has run and are pinned to
:data:`~data.schema.SCHEMA_VERSION` (Requirement 4.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from data.schema import SCHEMA_VERSION, DecisionRecord, Domain, options
from features.encoders import OneHotEncoder
from features.history import HistoryBuilder
from features.temporal import (
    RollingFrequencyTracker,
    dow_onehot,
    hour_cyclical,
    temporal_width,
    tod_onehot,
)
from personalization.profile_store import EMBEDDING_DIM

#: History window length (number of most recent decisions used as sequence).
DEFAULT_K = 5

#: A single engineered feature vector: a 1-D float array of width ``flat_dim``.
FeatureVector = np.ndarray


@dataclass
class FeatureMatrix:
    """Engineered features for a set of records in one domain."""

    X: np.ndarray  # (n, flat_dim) float
    seq: np.ndarray  # (n, K, step_dim) float
    y: list[str]  # actual decision label per row
    timestamps: list[datetime]
    contexts: list[dict]  # raw context dicts, for reporting

    def Xy(self) -> tuple[np.ndarray, list[str]]:
        return self.X, self.y


class FeaturePipeline:
    """Deterministic feature builder for a single decision domain."""

    def __init__(self, domain: str | Domain, k: int = DEFAULT_K) -> None:
        self.domain = Domain(domain) if not isinstance(domain, Domain) else domain
        self.k = k
        self.options = list(options(self.domain))
        self.loc_enc = OneHotEncoder()
        self.weather_enc = OneHotEncoder()
        self.stress_enc = OneHotEncoder()
        self.loc_tod_enc = OneHotEncoder()
        self._fitted = False
        #: Schema version the engineered layout is pinned to (Requirement 4.5).
        self.schema_version = SCHEMA_VERSION
        #: Fixed width of the per-user embedding block appended to every vector.
        self.embedding_dim = EMBEDDING_DIM
        # dimensions filled in during fit
        self.context_dim = 0
        self.flat_dim = 0
        self.step_dim = 0
        self.history_dim = 0

    # ------------------------------------------------------------------
    def fit(self, records: list[DecisionRecord]) -> "FeaturePipeline":
        """Learn categorical vocabularies from ``records`` of this domain."""
        dom_records = [r for r in records if r.domain == self.domain.value]
        self.loc_enc.fit([r.location for r in dom_records] or ["home"])
        self.weather_enc.fit([r.weather for r in dom_records] or ["clear"])
        self.stress_enc.fit([getattr(r, "stress_level", "medium") for r in dom_records] or ["medium"])
        self.loc_tod_enc.fit([f"{r.location}_{r.time_of_day}" for r in dom_records] or ["home_morning"])
        # context = location + weather + stress + loc_tod_interaction + day_type(2) + mood_energy(1)
        self.context_dim = self.loc_enc.width + self.weather_enc.width + self.stress_enc.width + self.loc_tod_enc.width + 2 + 1
        rolling_w = len(self.options)
        hist = HistoryBuilder(self.domain, self.k)
        self.history_dim = hist.flat_width
        self.flat_dim = (
            temporal_width()
            + rolling_w
            + self.context_dim
            + self.history_dim
            + self.embedding_dim
        )
        # per-step sequence dim: history one-hot slot + context (broadcast) is heavy;
        # keep step as [decision_onehot(vocab) + temporal(per record, repeated)].
        self.step_dim = hist.vocab_size
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def _context_vec(self, record_or_ctx) -> list[float]:
        loc = record_or_ctx.location
        weather = record_or_ctx.weather
        day_type = record_or_ctx.day_type
        time_of_day = record_or_ctx.time_of_day
        mood = float(record_or_ctx.mood_energy)
        stress = getattr(record_or_ctx, "stress_level", "medium")
        vec: list[float] = []
        vec.extend(self.loc_enc.transform_one(loc))
        vec.extend(self.weather_enc.transform_one(weather))
        vec.extend(self.stress_enc.transform_one(stress))
        vec.extend(self.loc_tod_enc.transform_one(f"{loc}_{time_of_day}"))
        vec.extend([1.0 if day_type == "weekday" else 0.0,
                    1.0 if day_type == "weekend" else 0.0])
        vec.append(mood)
        return vec

    def _temporal_vec(self, ts: datetime, time_of_day: str) -> list[float]:
        return [*hour_cyclical(ts), *dow_onehot(ts), *tod_onehot(time_of_day)]

    def _embedding_vec(self, profile) -> list[float]:
        """Return a fixed-width per-user embedding block.

        Pulls ``profile.embedding`` when a profile is supplied and falls back to a
        zero vector otherwise (e.g. cold-start or training without a profile). The
        result is always exactly :attr:`embedding_dim` long so the flat width stays
        stable across fit / transform / transform_one (Requirement 4.5).
        """
        emb = getattr(profile, "embedding", None) if profile is not None else None
        if emb is None:
            return [0.0] * self.embedding_dim
        vec = [float(v) for v in emb][: self.embedding_dim]
        if len(vec) < self.embedding_dim:
            vec = vec + [0.0] * (self.embedding_dim - len(vec))
        return vec

    # ------------------------------------------------------------------
    def transform(
        self, records: list[DecisionRecord], profile=None
    ) -> FeatureMatrix:
        """Transform domain records into flat + sequence features (no leakage).

        For the record at index ``i`` the rolling-frequency and last-K history
        components reflect only decisions strictly earlier than ``i`` (Requirements
        4.3, 4.4): the per-domain trackers are read *before* and updated *after* each
        record. ``profile`` supplies the per-user embedding block appended to every
        flat vector; when omitted a zero block of width :attr:`embedding_dim` is used
        so the output width is identical with or without a profile.
        """
        if not self._fitted:
            raise RuntimeError("FeaturePipeline.transform called before fit")
        dom_records = [r for r in records if r.domain == self.domain.value]
        dom_records = sorted(dom_records, key=lambda r: r.timestamp)

        hist = HistoryBuilder(self.domain, self.k)
        freq = RollingFrequencyTracker(self.domain)
        embedding = self._embedding_vec(profile)

        X_rows: list[list[float]] = []
        seq_rows: list[list[list[float]]] = []
        y: list[str] = []
        timestamps: list[datetime] = []
        contexts: list[dict] = []

        for r in dom_records:
            temporal = self._temporal_vec(r.timestamp, r.time_of_day)
            rolling = freq.features()
            context = self._context_vec(r)
            window_ids = hist.current()
            history_flat = hist.onehot(window_ids)

            X_rows.append([*temporal, *rolling, *context, *history_flat, *embedding])
            # sequence: one row per history step (one-hot of decision id)
            seq_rows.append([self._step_onehot(hist.vocab_size, i) for i in window_ids])

            y.append(r.decision_made)
            timestamps.append(r.timestamp)
            contexts.append({
                "location": r.location, "weather": r.weather,
                "day_type": r.day_type, "time_of_day": r.time_of_day,
                "mood_energy": r.mood_energy, "stress_level": getattr(r, "stress_level", "medium")
            })

            # update AFTER reading features (prevents future leakage)
            hist.update(r.decision_made)
            freq.update(r.decision_made)

        X = np.asarray(X_rows, dtype=np.float32) if X_rows else np.zeros((0, self.flat_dim), np.float32)
        seq = (np.asarray(seq_rows, dtype=np.float32)
               if seq_rows else np.zeros((0, self.k, self.step_dim), np.float32))
        return FeatureMatrix(X=X, seq=seq, y=y, timestamps=timestamps, contexts=contexts)

    @staticmethod
    def _step_onehot(vocab_size: int, idx: int) -> list[float]:
        slot = [0.0] * vocab_size
        slot[idx] = 1.0
        return slot

    # ------------------------------------------------------------------
    def transform_one(
        self,
        context,
        recent: list[DecisionRecord],
        profile=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a single flat vector + sequence tensor for the prediction path.

        ``context`` must expose ``location``, ``weather``, ``day_type``,
        ``time_of_day``, ``mood_energy`` and a ``timestamp``. ``recent`` is the
        time-ordered recent decision history for this domain (may be empty / shorter
        than K; it is left-padded). ``profile`` supplies the per-user embedding block;
        when ``None`` a zero block is used so the width matches :meth:`transform`.

        Returns a ``(1, flat_dim)`` flat vector and a ``(1, K, step_dim)`` sequence
        tensor, ready for the baseline and sequence models respectively.
        """
        if not self._fitted:
            raise RuntimeError("FeaturePipeline.transform_one called before fit")
        hist = HistoryBuilder(self.domain, self.k)
        freq = RollingFrequencyTracker(self.domain)
        for r in sorted(recent, key=lambda x: x.timestamp):
            if r.domain == self.domain.value:
                hist.update(r.decision_made)
                freq.update(r.decision_made)

        ts = getattr(context, "timestamp", None) or datetime.now()
        temporal = self._temporal_vec(ts, context.time_of_day)
        rolling = freq.features()
        ctx = self._context_vec(context)
        window_ids = hist.current()
        history_flat = hist.onehot(window_ids)
        embedding = self._embedding_vec(profile)

        x = np.asarray([[*temporal, *rolling, *ctx, *history_flat, *embedding]],
                       dtype=np.float32)
        seq = np.asarray([[self._step_onehot(hist.vocab_size, i) for i in window_ids]],
                         dtype=np.float32)
        return x, seq
