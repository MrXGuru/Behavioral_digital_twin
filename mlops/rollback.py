"""Rollback controller: promote only if candidate doesn't underperform baseline."""
from __future__ import annotations
import logging
from typing import Union

from mlops.registry import ModelRegistry, ModelVersion

logger = logging.getLogger(__name__)


class RollbackController:
    """Evaluates a candidate model version against a baseline.

    A candidate is kept (promoted) only when it does **not** underperform the baseline
    within tolerance across all three metrics:

    * ``accuracy``  ≥ baseline accuracy − tol
    * ``macro_f1``  ≥ baseline macro_f1 − tol
    * ``brier``     ≤ baseline brier + tol  (lower is better)

    If the candidate fails this check the controller rolls back to the most recent
    prior version via :meth:`ModelRegistry.previous`.

    :param registry: The :class:`ModelRegistry` instance to use for promotion/rollback.
    :param tol: Tolerance within which the candidate may underperform (default 0.05).
    """

    def __init__(self, registry: ModelRegistry, tol: float = 0.05) -> None:
        self.registry = registry
        self.tol = tol

    def evaluate(
        self,
        domain: str,
        candidate: Union[ModelVersion, dict],
        baseline: Union[ModelVersion, dict],
    ) -> bool:
        """Return True (keep candidate) if it doesn't underperform baseline within tol.

        Accepts either :class:`ModelVersion` objects (from which ``.metrics`` is
        extracted) or plain ``dict`` objects directly so callers can pass either form.

        Promotes only if:
        * ``accuracy``  ≥ baseline accuracy  − tol
        * ``macro_f1``  ≥ baseline macro_f1  − tol
        * ``brier``     ≤ baseline brier     + tol
        """
        c = candidate.metrics if isinstance(candidate, ModelVersion) else candidate
        b = baseline.metrics if isinstance(baseline, ModelVersion) else baseline
        acc_ok = c.get("accuracy", 0) >= b.get("accuracy", 0) - self.tol
        f1_ok = c.get("macro_f1", 0) >= b.get("macro_f1", 0) - self.tol
        brier_ok = c.get("brier", 1) <= b.get("brier", 1) + self.tol
        keep = acc_ok and f1_ok and brier_ok
        if not keep:
            logger.warning(
                "rollback triggered domain=%s",
                domain,
            )
        return keep

    def rollback(self, domain: str) -> ModelVersion | None:
        """Restore the previous model version for *domain*.

        Returns the restored :class:`ModelVersion`, or *None* if there is no prior
        version to roll back to.
        """
        prev = self.registry.previous(domain)
        if prev:
            self.registry.promote(prev)
            logger.info(
                "rolled_back domain=%s to version=%s", domain, prev.version
            )
        return prev
