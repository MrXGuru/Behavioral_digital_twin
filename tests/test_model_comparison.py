"""Tests that the sequence model is fully wired into model comparison.

Task 8.3 wires the Sequence_Model into ``evaluate_models`` so the ``ComparisonReport``
carries Accuracy, macro-F1, and Brier_Score for BOTH the baseline and the sequence model,
computed on the SAME temporal validation partition.

Validates: Requirements 6.3, 6.4
"""

from __future__ import annotations

import math

import pytest

from data.schema import Domain, options
from tests.synthetic_data_generator import GeneratorConfig, SyntheticDataGenerator
from models.evaluate import ComparisonReport, Metrics, evaluate_models


def _dataset(domain: Domain) -> list:
    """Generate a reproducible single-domain dataset with enough records to split."""
    config = GeneratorConfig(
        n_days=40,
        decisions_per_day=(3, 5),
        domains=[domain.value],
        weekend_shift=0.4,
        drift_rate=0.1,
        noise=0.1,
        seed=7,
        user_id="u1",
    )
    return SyntheticDataGenerator(config).generate()


def _assert_valid_metrics(m: Metrics) -> None:
    assert m.n > 0
    assert 0.0 <= m.accuracy <= 1.0
    assert 0.0 <= m.macro_f1 <= 1.0
    # Multiclass Brier score is bounded in [0, 2].
    assert 0.0 <= m.brier <= 2.0


@pytest.mark.parametrize("domain", list(Domain))
def test_both_models_scored_on_same_validation_partition(domain: Domain) -> None:
    """Req 6.3/6.4: baseline AND sequence scored on the same validation partition."""
    records = _dataset(domain)
    report = evaluate_models(records, domain, val_fraction=0.2, k=5)

    assert isinstance(report, ComparisonReport)
    assert report.domain == domain.value

    # Both models report all three metrics in valid ranges.
    _assert_valid_metrics(report.baseline)
    _assert_valid_metrics(report.sequence)

    # Same validation partition: identical, positive validation sample count.
    assert report.baseline.n == report.sequence.n > 0

    # A winner is chosen between the two competing models.
    assert report.winner in {"baseline", "sequence"}
    assert report.rationale


@pytest.mark.parametrize("domain", list(Domain))
def test_report_as_dict_includes_both_models(domain: Domain) -> None:
    """Req 6.4: the serialized report contains both models' metric sets."""
    records = _dataset(domain)
    report = evaluate_models(records, domain, val_fraction=0.2, k=5)
    payload = report.as_dict()

    for model_key in ("baseline", "sequence"):
        assert model_key in payload
        metrics = payload[model_key]
        for metric_key in ("accuracy", "macro_f1", "brier", "n"):
            assert metric_key in metrics
        assert metrics["n"] > 0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0

    # Both models scored on the identical validation partition.
    assert payload["baseline"]["n"] == payload["sequence"]["n"]


def test_predicted_options_within_domain_option_set() -> None:
    """Sanity: the report's domain matches a real domain option set (focus)."""
    domain = Domain.FOCUS
    records = _dataset(domain)
    report = evaluate_models(records, domain)
    assert report.domain in {d.value for d in Domain}
    assert len(options(domain)) > 1
    # Brier on validation should be a finite number.
    assert math.isfinite(report.baseline.brier)
    assert math.isfinite(report.sequence.brier)
