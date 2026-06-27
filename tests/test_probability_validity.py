"""Property-based test for probability validity of the decision models.

Property 5: Probability validity
    *For any* prediction from either the :class:`~models.baseline.BaselineModel` or the
    :class:`~models.sequence.SequenceModel`, ``predict_proba`` returns a valid
    distribution over the domain's option set: every value is non-negative, the values
    sum to ~1.0, and the keys are exactly the domain option set. Additionally
    ``predict() == argmax(class_probs)`` and ``confidence() == max(class_probs)``, and the
    predicted decision is a member of the domain's option set.

**Validates: Requirements 5.3, 5.4, 7.2, 7.3**
"""

from __future__ import annotations

import math

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from data.schema import Domain, options
from tests.synthetic_data_generator import GeneratorConfig, SyntheticDataGenerator
from features.feature_pipeline import FeaturePipeline
from models.base import DecisionModel, is_valid_distribution
from models.baseline import BaselineModel
from models.sequence import SequenceModel

# All supported domain string values.
_SUPPORTED_DOMAINS = [d.value for d in Domain]


@st.composite
def domain_and_config(draw: st.DrawFn) -> tuple[str, GeneratorConfig]:
    """Draw a single domain plus a small, always-valid ``GeneratorConfig`` for it.

    The config is restricted to the drawn domain and kept modest (few days, a small
    decisions-per-day range) because both models are trained once per example. Staying
    inside the generator's precondition space guarantees the config is acceptable.
    """
    domain = draw(st.sampled_from(_SUPPORTED_DOMAINS))

    n_days = draw(st.integers(min_value=5, max_value=10))
    lo = draw(st.integers(min_value=1, max_value=3))
    hi = draw(st.integers(min_value=lo, max_value=lo + 3))

    unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    weekend_shift = draw(unit)
    drift_rate = draw(unit)
    # Keep noise modest so the data has learnable structure but stays varied.
    noise = draw(st.floats(min_value=0.0, max_value=0.5, allow_nan=False,
                           allow_infinity=False))

    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    config = GeneratorConfig(
        n_days=n_days,
        decisions_per_day=(lo, hi),
        domains=[domain],
        weekend_shift=weekend_shift,
        drift_rate=drift_rate,
        noise=noise,
        seed=seed,
    )
    return domain, config


def _assert_valid_prediction(
    model: DecisionModel, domain: str, x, seq
) -> None:
    """Assert Property 5 holds for one model on one prepared sample."""
    option_set = set(options(domain))

    probs = model.predict_proba(x, seq)

    # Keys are exactly the domain option set.
    assert set(probs.keys()) == option_set, (
        f"{model.name}: keys {set(probs.keys())} != options {option_set}"
    )
    # Every value non-negative.
    assert all(v >= 0.0 for v in probs.values()), (
        f"{model.name}: negative probability in {probs}"
    )
    # Values sum to ~1.0 (and pass the shared distribution validity helper).
    total = sum(probs.values())
    assert math.isclose(total, 1.0, abs_tol=1e-6), (
        f"{model.name}: probabilities sum to {total}, expected ~1.0"
    )
    assert is_valid_distribution(probs), (
        f"{model.name}: not a valid distribution: {probs}"
    )

    # predict() == argmax(class_probs) and is in the option set (Req 5.4 / 7.3).
    predicted = model.predict(x, seq)
    expected_argmax = max(probs, key=probs.get)
    assert predicted == expected_argmax, (
        f"{model.name}: predict() {predicted!r} != argmax {expected_argmax!r}"
    )
    assert predicted in option_set, (
        f"{model.name}: predicted {predicted!r} not in option set {option_set}"
    )

    # confidence() == max(class_probs).
    confidence = model.confidence(x, seq)
    assert math.isclose(confidence, max(probs.values()), abs_tol=1e-9), (
        f"{model.name}: confidence {confidence} != max prob {max(probs.values())}"
    )


@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(spec=domain_and_config())
def test_predict_proba_is_valid_distribution(spec: tuple[str, GeneratorConfig]) -> None:
    """Both models yield a valid distribution and consistent predict/confidence.

    Generates data for one domain, fits a single feature pipeline, fits BOTH models on
    the same engineered features (the sequence model with a small epoch budget to keep
    training fast), then checks Property 5 over several validation rows for each model.
    """
    domain, config = spec

    records = SyntheticDataGenerator(config).generate()

    pipeline = FeaturePipeline(domain).fit(records)
    matrix = pipeline.transform(records)
    n = matrix.X.shape[0]
    assert n > 0, "expected at least one engineered row for the domain"

    # Fit both models on the same (X, seq, y). Keep the sequence model fast.
    baseline = BaselineModel(pipeline.options).fit(matrix.X, matrix.seq, matrix.y)
    sequence = SequenceModel(pipeline.options, epochs=5).fit(
        matrix.X, matrix.seq, matrix.y
    )

    # Check several validation rows (sample a handful spread across the matrix).
    row_indices = sorted({0, n // 2, n - 1, min(1, n - 1), max(0, n - 2)})
    for model in (baseline, sequence):
        for i in row_indices:
            x = matrix.X[i : i + 1]
            seq = matrix.seq[i : i + 1]
            _assert_valid_prediction(model, domain, x, seq)
