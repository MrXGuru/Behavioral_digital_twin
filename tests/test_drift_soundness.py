"""Property-based test for concept-drift flag soundness.

Property 8: Drift flag soundness
    For any sequence of ``record(predicted, actual, confidence)`` calls on a
    ``DriftDetector(window, threshold)``, the ``status()`` result satisfies:

    - ``drift`` is True if and only if the number of labeled predictions in the rolling
      window is at least ``window`` AND ``window_acc < threshold``. (Req 10.3)
    - When there is at least one labeled prediction, ``window_acc`` equals the proportion
      of correct (``predicted == actual``) predictions over the most recent ``window``
      labeled predictions. (Req 10.4)
    - When there are zero labeled predictions, ``window_acc`` is None and ``drift`` is
      False. (Req 10.2, supporting)

Validates: Requirements 10.3, 10.4
"""

from __future__ import annotations

import math

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from api.drift import DriftDetector

#: Small label space so that correct matches and collisions both occur frequently.
_LABELS = ["a", "b", "c"]


@st.composite
def drift_scenarios(draw: st.DrawFn) -> tuple[int, float, list[tuple[str, str | None, float]]]:
    """Draw a ``(window, threshold, events)`` scenario.

    Events are ``(predicted, actual_or_None, confidence)`` triples drawn from a small label
    space so that correct/incorrect predictions occur with reasonable frequency. Some
    events are unlabeled (``actual is None``) to exercise the rule that only labeled
    predictions enter the rolling window. The number of events is allowed to range above
    and below ``window`` so both the "fewer than window" and "full window" regimes are hit.
    """
    window = draw(st.integers(min_value=1, max_value=30))
    threshold = draw(st.floats(min_value=0.0, max_value=1.0))
    events = draw(
        st.lists(
            st.tuples(
                st.sampled_from(_LABELS),
                st.one_of(st.none(), st.sampled_from(_LABELS)),
                st.floats(min_value=0.0, max_value=1.0),
            ),
            min_size=0,
            max_size=60,
        )
    )
    return window, threshold, events


@given(scenario=drift_scenarios())
@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
def test_drift_flag_soundness(
    scenario: tuple[int, float, list[tuple[str, str | None, float]]],
) -> None:
    """``status()`` matches an independent recomputation of window_acc and drift."""
    window, threshold, events = scenario

    detector = DriftDetector(window=window, threshold=threshold)
    for predicted, actual, confidence in events:
        detector.record(predicted, actual, confidence)

    status = detector.status()

    # Independently recompute from the labeled subset: only the most recent `window`
    # labeled events occupy the rolling window.
    labeled = [(p, a) for (p, a, _c) in events if a is not None]
    recent = labeled[-window:]
    observed = len(recent)

    if observed == 0:
        # Req 10.2 (supporting): no labeled predictions -> window_acc None, no drift.
        assert status.window_acc is None
        assert status.drift is False
        return

    correct = sum(1 for p, a in recent if p == a)
    expected_acc = correct / observed

    # Req 10.4: window_acc is the proportion correct over the most recent `window` labeled.
    assert status.window_acc is not None
    assert math.isclose(status.window_acc, expected_acc, rel_tol=1e-9, abs_tol=1e-12)

    # Req 10.3: drift iff at least `window` labeled predictions exist AND acc < threshold.
    expected_drift = observed >= window and expected_acc < threshold
    assert status.drift is expected_drift

    # score is defined as 1 - window_acc.
    assert math.isclose(status.score, 1.0 - expected_acc, rel_tol=1e-9, abs_tol=1e-12)
