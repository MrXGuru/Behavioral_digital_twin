"""Unit tests for the baseline model save/load round-trip.

These tests verify Requirement 5.5: a model persisted with ``save()`` and reloaded with
``load()`` yields predictions equal to the original model for identical inputs. Both the
trained-classifier path (gradient boosting over the flat features) and the degenerate
single-class fallback path are covered.

Validates: Requirements 5.5
"""

from __future__ import annotations

import numpy as np
import pytest

from data.schema import Domain, options
from tests.synthetic_data_generator import GeneratorConfig, SyntheticDataGenerator
from features.feature_pipeline import FeaturePipeline
from models.baseline import BaselineModel

#: Domain used for the single-domain dataset in these tests.
_DOMAIN = Domain.FOCUS


def _build_features() -> tuple[FeaturePipeline, np.ndarray, np.ndarray, list[str]]:
    """Generate a small single-domain dataset and engineer features for it.

    Returns the fitted pipeline plus the flat matrix ``X``, sequence tensor ``seq`` and
    labels ``y`` for the FOCUS domain. The generator config is fixed (seeded) so the
    dataset is reproducible, with enough volume and noise that more than one focus
    option appears (exercising the real-classifier path, not the fallback).
    """
    config = GeneratorConfig(
        n_days=40,
        decisions_per_day=(2, 4),
        domains=[_DOMAIN.value],
        weekend_shift=0.5,
        drift_rate=0.2,
        noise=0.3,
        seed=7,
    )
    records = SyntheticDataGenerator(config).generate()

    pipeline = FeaturePipeline(_DOMAIN, k=5).fit(records)
    fm = pipeline.transform(records)
    return pipeline, fm.X, fm.seq, fm.y


def _assert_round_trip_equal(
    original: BaselineModel,
    loaded: BaselineModel,
    X: np.ndarray,
    seq: np.ndarray,
) -> None:
    """Assert the loaded model matches the original for every row in ``X``."""
    n = X.shape[0]
    assert n > 0, "expected a non-empty feature matrix to compare predictions over"
    for i in range(n):
        x_row = X[i : i + 1]
        seq_row = seq[i : i + 1]

        orig_probs = original.predict_proba(x_row, seq_row)
        loaded_probs = loaded.predict_proba(x_row, seq_row)

        # Per-option probabilities are equal within floating-point tolerance.
        assert orig_probs.keys() == loaded_probs.keys()
        for opt in original.options:
            assert loaded_probs[opt] == pytest.approx(orig_probs[opt])

        # argmax prediction matches as well (Requirement 5.5).
        assert loaded.predict(x_row, seq_row) == original.predict(x_row, seq_row)


def test_save_load_round_trip_trained_classifier(tmp_path) -> None:
    """A saved-then-loaded trained model predicts identically to the original."""
    pipeline, X, seq, y = _build_features()

    model = BaselineModel(pipeline.options).fit(X, seq, y)
    # Sanity: this dataset should exercise the real-classifier path, not the fallback.
    assert model._clf is not None
    assert len(set(y)) >= 2

    path = tmp_path / "baseline.pkl"
    model.save(str(path))
    loaded = BaselineModel.load(str(path))

    # The reloaded model preserves the option set it was trained on.
    assert loaded.options == model.options

    _assert_round_trip_equal(model, loaded, X, seq)


def test_save_load_round_trip_single_class_fallback(tmp_path) -> None:
    """The single-class fallback path also round-trips to identical predictions."""
    pipeline, X, seq, _ = _build_features()

    # Force the degenerate single-class case: every label is the same option.
    single_label = options(_DOMAIN)[0]
    y_single = [single_label] * X.shape[0]

    model = BaselineModel(pipeline.options).fit(X, seq, y_single)
    # Sanity: a single class must trigger the constant-prior fallback (no classifier).
    assert model._clf is None
    assert model._fallback_probs is not None

    path = tmp_path / "baseline_single_class.pkl"
    model.save(str(path))
    loaded = BaselineModel.load(str(path))

    assert loaded.options == model.options
    # The fallback prior is preserved verbatim across the round-trip.
    assert loaded._fallback_probs == model._fallback_probs

    _assert_round_trip_equal(model, loaded, X, seq)
