"""Online per-user profile updates (EMA embedding + monotonic counts).

:class:`ProfileUpdater` applies an incremental, side-effect-free update to a
:class:`~personalization.profile_store.UserProfile` as new decision records arrive,
without any full retraining. ``ProfileUpdater.update``:

* **Counts (Req 11.2)** — increments ``decision_counts[decision_made]`` by the number
  of occurrences of that option in the new records. Counts only ever increase, so no
  count can drop below its prior value (monotonic non-decreasing).
* **Embedding (Req 11.3)** — moves the user embedding toward the aggregate behavior of
  the new records via an exponential moving average (EMA):
  ``new = (1 - alpha) * old + alpha * aggregate``.
* **Timestamp (Req 11.4)** — advances ``last_updated`` to the maximum timestamp among
  the new records, but never moves it backwards: if every new record is older than the
  current ``last_updated`` the existing (larger) value is kept.

Aggregate behavior vector
-------------------------
The aggregate vector is a fixed-length (``EMBEDDING_DIM``) summary of *which* options
were chosen in the new records. Each option string is mapped to a bin index in
``[0, EMBEDDING_DIM)`` with a stable (process-independent) hash; every new record adds
one occurrence to its option's bin, and the bins are L1-normalized so the vector sums to
one (or is all-zero when there are no records). Hashing — rather than truncating a global
option ordering — guarantees that *every* option, across all domains, contributes to the
summary instead of silently dropping options beyond the embedding length.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from data.schema import DecisionRecord
from personalization.profile_store import (
    EMBEDDING_DIM,
    UserProfile,
)

# ``EMBED_DIM`` is a backwards-compatible alias of ``EMBEDDING_DIM``; both are exported by
# ``profile_store``. The updater uses the canonical name and re-exports the alias so older
# imports of ``EMBED_DIM`` from this module keep working.
EMBED_DIM: int = EMBEDDING_DIM

#: Default EMA learning rate (the weight given to the new records' aggregate behavior).
DEFAULT_LEARNING_RATE: float = 0.3


def _as_aware_utc(ts: datetime) -> datetime:
    """Return ``ts`` as a timezone-aware UTC datetime.

    Datetimes flow into the updater from two sources that historically disagreed on
    awareness: the :class:`~personalization.profile_store.UserProfile` cold-start
    ``last_updated`` is tz-aware UTC, while the synthetic data generator (and other
    callers) emit *naive* timestamps. Comparing the two directly raises
    ``TypeError: can't compare offset-naive and offset-aware datetimes``.

    Normalizing every timestamp through this helper makes the comparison total: a
    naive timestamp is interpreted as UTC, and an already-aware timestamp is returned
    unchanged. Aware-only flows (e.g. the monotonicity tests) are therefore unaffected.
    """
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def _option_bin(option: str, dim: int = EMBEDDING_DIM) -> int:
    """Map an option string to a stable bin index in ``[0, dim)``.

    Uses a content hash (BLAKE2b) so the mapping is deterministic across processes and
    runs, unlike Python's built-in :func:`hash` for strings.
    """
    digest = hashlib.blake2b(option.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _aggregate_vector(records: list[DecisionRecord],
                      dim: int = EMBEDDING_DIM) -> list[float]:
    """Return an L1-normalized, fixed-``dim`` summary of the new records' decisions.

    Each record contributes one occurrence to the bin of its ``decision_made`` option.
    The returned vector sums to one when ``records`` is non-empty and is the all-zero
    vector otherwise.
    """
    bins = [0.0] * dim
    for record in records:
        bins[_option_bin(record.decision_made, dim)] += 1.0
    total = sum(bins)
    if total == 0.0:
        return bins
    return [count / total for count in bins]


class ProfileUpdater:
    """Apply incremental, monotonic per-user profile updates.

    Args:
        learning_rate: EMA weight ``alpha`` in ``(0, 1]`` applied to the new records'
            aggregate behavior. Larger values adapt faster to recent behavior.
    """

    def __init__(self, learning_rate: float = DEFAULT_LEARNING_RATE) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        self.learning_rate = learning_rate

    def update(self, profile: UserProfile,
               new_records: list[DecisionRecord]) -> UserProfile:
        """Fold ``new_records`` into ``profile`` and return the updated profile.

        The update is side-effect free: a new :class:`UserProfile` is returned and the
        input ``profile`` is left unchanged. ``user_id`` is preserved. When
        ``new_records`` is empty the original ``profile`` is returned unchanged.

        Guarantees:
            * each ``decision_counts`` value is >= its prior value (Req 11.2);
            * the embedding is an EMA blend of the old embedding and the new records'
              aggregate behavior (Req 11.3);
            * ``last_updated`` is non-decreasing and equals
              ``max(existing, max(record.timestamp for record in new_records))``
              (Req 11.4).
        """
        if not new_records:
            return profile

        # Req 11.2: increment each option's count by its occurrences (monotonic).
        new_counts = dict(profile.decision_counts)
        for record in new_records:
            new_counts[record.decision_made] = (
                new_counts.get(record.decision_made, 0) + 1
            )

        # Req 11.3: EMA the embedding toward the aggregate behavior of the new records.
        aggregate = _aggregate_vector(new_records, len(profile.embedding))
        alpha = self.learning_rate
        new_embedding = [
            (1.0 - alpha) * old + alpha * agg
            for old, agg in zip(profile.embedding, aggregate)
        ]

        # Req 11.4: advance last_updated to the max new timestamp, never backwards.
        # Compare on a common tz-aware (UTC) basis so naive timestamps (e.g. from the
        # synthetic generator) and the tz-aware cold-start ``last_updated`` are
        # comparable; the stored value is likewise normalized to tz-aware UTC.
        prior_last_updated = _as_aware_utc(profile.last_updated)
        max_new_ts = _as_aware_utc(max(record.timestamp for record in new_records))
        new_last_updated = (
            max_new_ts if max_new_ts > prior_last_updated else prior_last_updated
        )

        return UserProfile(
            user_id=profile.user_id,
            embedding=new_embedding,
            decision_counts=new_counts,
            last_updated=new_last_updated,
        )
