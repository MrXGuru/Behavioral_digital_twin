"""Per-user behavioral profile storage.

A :class:`UserProfile` holds a small per-user embedding (an aggregate behavior
vector), per-option decision counts, and a ``last_updated`` timestamp. The
:class:`UserProfileStore` provides ``get``/``upsert`` access and, crucially,
returns a deterministic cold-start default for unknown users so the prediction
path never fails on a first-time user.

The embedding dimension is fixed by the module constant :data:`EMBEDDING_DIM`
so cold-start defaults are consistent and :class:`~personalization.updater.ProfileUpdater`
can rely on a stable vector length. ``EMBED_DIM`` is kept as a backwards-compatible
alias of the same value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

#: Fixed dimension of every per-user behavioral embedding. Chosen once here so the
#: cold-start default, the online updater, and any consumer all agree on the length.
EMBEDDING_DIM: int = 8

#: Backwards-compatible alias for :data:`EMBEDDING_DIM`.
EMBED_DIM: int = EMBEDDING_DIM

#: Sensible cold-start ``last_updated`` for a user we have never seen. Using the Unix
#: epoch (UTC) keeps the value a real :class:`datetime` (never ``None``) and ensures
#: the updater's non-decreasing ``last_updated`` rule holds against any real timestamp.
COLD_START_TIMESTAMP: datetime = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass
class UserProfile:
    """A per-user behavioral profile.

    Attributes:
        user_id: Identifier of the user this profile belongs to.
        embedding: Learned/aggregated behavioral vector of length
            :data:`EMBEDDING_DIM`.
        decision_counts: Per domain/class frequencies (option -> count).
        last_updated: Timestamp of the most recent record folded into the profile;
            :data:`COLD_START_TIMESTAMP` for a freshly created cold-start profile.
    """

    user_id: str
    embedding: list[float] = field(default_factory=lambda: [0.0] * EMBEDDING_DIM)
    decision_counts: dict[str, int] = field(default_factory=dict)
    last_updated: datetime = COLD_START_TIMESTAMP

    @classmethod
    def cold_start(cls, user_id: str) -> "UserProfile":
        """Build a deterministic cold-start profile for an unknown user.

        The returned profile has a zero embedding of length :data:`EMBEDDING_DIM`,
        empty decision counts, and ``last_updated`` set to
        :data:`COLD_START_TIMESTAMP`.
        """
        return cls(
            user_id=user_id,
            embedding=[0.0] * EMBEDDING_DIM,
            decision_counts={},
            last_updated=COLD_START_TIMESTAMP,
        )

    def embedding_summary(self) -> dict:
        """Return a compact, JSON-friendly summary of the embedding."""
        return {
            "dim": len(self.embedding),
            "norm": round(sum(v * v for v in self.embedding) ** 0.5, 4),
        }


class UserProfileStore:
    """In-memory per-user profile store with cold-start defaults.

    The store is intentionally backed by a simple dict for now; the ``get``/``upsert``
    contract is stable so the backing store can later be swapped for a persistent one
    without touching callers.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, UserProfile] = {}

    def get(self, user_id: str) -> UserProfile:
        """Return the stored profile for ``user_id``.

        For an unknown user this returns a fresh cold-start default (see
        :meth:`UserProfile.cold_start`) and never raises. The cold-start profile is
        not persisted; call :meth:`upsert` to store it.
        """
        profile = self._profiles.get(user_id)
        if profile is None:
            return UserProfile.cold_start(user_id)
        return profile

    def upsert(self, profile: UserProfile) -> None:
        """Insert or replace the stored profile for ``profile.user_id``."""
        self._profiles[profile.user_id] = profile

    def has(self, user_id: str) -> bool:
        """Return ``True`` if a profile is stored for ``user_id``."""
        return user_id in self._profiles

    def delete(self, user_id: str) -> None:
        """Remove the stored profile for ``user_id`` if present (no-op otherwise)."""
        self._profiles.pop(user_id, None)
