"""Last-K decision-sequence builder.

Builds the behavioral-history component of a feature vector: the ids of the last ``K``
decisions for a domain, left-padded with the reserved ``PAD`` id. The window is updated
only *after* features for the current record are produced, guaranteeing that history at
index ``i`` reflects only decisions strictly earlier than ``i`` (no future leakage).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

from data.schema import PAD, Domain, options


class HistoryBuilder:
    """Maintain and emit the last-K decision-id window for a single domain.

    Decision options are mapped to integer ids ``1..n``; the ``PAD`` sentinel is id ``0``.
    """

    def __init__(self, domain: str | Domain, k: int) -> None:
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        self.k = k
        self.options = list(options(domain))
        # id 0 reserved for PAD; options get 1..n
        self._id = {PAD: 0}
        for i, opt in enumerate(self.options, start=1):
            self._id[opt] = i
        self.vocab_size = len(self.options) + 1  # + PAD
        self._window: deque[int] = deque(maxlen=k)

    def current(self) -> list[int]:
        """Return the current last-K window, left-padded with PAD id (0)."""
        ids = list(self._window)
        pad = [0] * (self.k - len(ids))
        return pad + ids

    def update(self, decision: str) -> None:
        """Push one decision id into the window."""
        self._window.append(self._id.get(decision, 0))

    def reset(self) -> None:
        """Clear the rolling window back to the cold-start (all-PAD) state."""
        self._window.clear()

    def id_of(self, decision: str) -> int:
        """Map a decision option to its integer id (PAD id ``0`` if unknown)."""
        return self._id.get(decision, 0)

    def sequences(self, decisions: Iterable[str]) -> list[list[int]]:
        """Build the per-record last-K id windows for ``decisions`` in order.

        For each decision at position ``i`` the emitted window holds the ids of the
        last ``K`` decisions strictly *before* ``i`` (records ``i-K .. i-1``),
        left-padded with the PAD id to exactly length ``K``; position ``0`` (cold
        start) yields an all-PAD window.

        The window is updated only *after* the current record's window is emitted,
        so a record's own decision never leaks into its own history (no future
        leakage). This rebuilds from a clean window and leaves the builder reset.
        """
        self.reset()
        windows: list[list[int]] = []
        for decision in decisions:
            windows.append(self.current())  # features BEFORE this record
            self.update(decision)  # update AFTER, never leaking the current record
        self.reset()
        return windows

    def transform_one(self, recent: Sequence[str]) -> list[int]:
        """Left-pad the last-K of ``recent`` decisions to a length-K id window.

        Prediction-path helper: ``recent`` is the time-ordered recent decision
        history for this domain (it may be empty or longer/shorter than ``K``).
        Only the last ``K`` entries are kept and the result is left-padded with the
        PAD id (0) to exactly length ``K``.
        """
        ids = [self._id.get(d, 0) for d in recent][-self.k :]
        pad = [0] * (self.k - len(ids))
        return pad + ids

    def onehot(self, ids: list[int]) -> list[float]:
        """Flatten a window of ids into a concatenated one-hot vector (K * vocab_size)."""
        vec: list[float] = []
        for i in ids:
            slot = [0.0] * self.vocab_size
            slot[i] = 1.0
            vec.extend(slot)
        return vec

    @property
    def flat_width(self) -> int:
        return self.k * self.vocab_size
