"""Unit tests for the sequence model: save/load round-trip and cold-start padding.

Covers the artifact round-trip equality required of every ``DecisionModel`` and the
prediction path over a padded (cold-start) history window.

* Round-trip equality (Requirement 5.5): a saved-then-loaded :class:`SequenceModel`
  yields predictions identical to the original for every input row -- exercised against
  the ACTIVE torch (LSTM) backend, which is installed in this environment.
* Cold-start padding: :meth:`FeaturePipeline.transform_one` with an empty ``recent``
  history produces an all-``PAD`` sequence; the model must still return a valid
  distribution over the option set without error.

Validates: Requirements 5.5
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.synthetic_data_generator import GeneratorConfig, SyntheticDataGenerator
from features.feature_pipeline import FeaturePipeline
from models.base import is_valid_distribution
from models.sequence import SequenceModel

DOMAIN = "focus"
SEED = 7


@pytest.fixture
def fitted_pipeline_and_features():
    """Generate a small single-domain dataset and fit a FeaturePipeline.

    Returns the fitted pipeline and the engineered :class:`FeatureMatrix`. The dataset
    is sized so the sequence model trains its real backend (>= 10 rows, >= 2 classes)
    rather than falling back to a label prior.
    """
    config = GeneratorConfig(
        n_days=12,
        decisions_per_day=(3, 5),
        domains=[DOMAIN],
        weekend_shift=0.3,
        drift_rate=0.1,
        noise=0.1,
        seed=SEED,
    )
    records = SyntheticDataGenerator(config).generate()

    pipeline = FeaturePipeline(DOMAIN, k=5)
    pipeline.fit(records)
    features = pipeline.transform(records)
    return pipeline, features


@pytest.fixture
def fitted_model(fitted_pipeline_and_features):
    """Fit a small SequenceModel on the engineered features (deterministic seed)."""
    import torch

    _, features = fitted_pipeline_and_features
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    options = list(set(features.y))
    # Build over the full domain option set so the label space is stable.
    from data.schema import options as domain_options

    model = SequenceModel(list(domain_options(DOMAIN)), epochs=5, hidden=8)
    model.fit(features.X, features.seq, features.y)
    return model


def test_active_backend_is_torch(fitted_model):
    """The active backend in this environment is the torch LSTM (sanity check)."""
    assert fitted_model.backend == "torch"
    # A real network was trained (not the label-prior fallback).
    assert fitted_model._net is not None
    assert fitted_model._fallback_probs is None


def test_save_load_roundtrip_predictions_equal(
    fitted_model, fitted_pipeline_and_features, tmp_path
):
    """Saved-then-loaded model matches the original prediction for every row (Req 5.5)."""
    _, features = fitted_pipeline_and_features
    path = str(tmp_path / "sequence_focus.pkl")

    fitted_model.save(path)
    loaded = SequenceModel.load(path)

    assert loaded.backend == fitted_model.backend
    assert loaded.options == fitted_model.options

    n = features.X.shape[0]
    assert n > 0
    for i in range(n):
        x = features.X[i : i + 1]
        seq = features.seq[i : i + 1]

        orig_probs = fitted_model.predict_proba(x, seq)
        loaded_probs = loaded.predict_proba(x, seq)

        assert orig_probs.keys() == loaded_probs.keys()
        for opt in orig_probs:
            assert loaded_probs[opt] == pytest.approx(orig_probs[opt], abs=1e-6)

        # predict() (argmax) must agree as well.
        assert loaded.predict(x, seq) == fitted_model.predict(x, seq)


def test_cold_start_padded_history_predicts_valid_distribution(
    fitted_model, fitted_pipeline_and_features
):
    """An all-PAD cold-start sequence yields a valid distribution without error."""
    pipeline, _ = fitted_pipeline_and_features

    from datetime import datetime
    from types import SimpleNamespace

    # Prediction-path context with NO recent history -> sequence is fully PAD-padded.
    context = SimpleNamespace(
        location="home",
        weather="clear",
        day_type="weekday",
        time_of_day="morning",
        mood_energy=0.6,
        timestamp=datetime(2024, 6, 1, 9, 0, 0),
    )
    x, seq = pipeline.transform_one(context, recent=[], profile=None)

    # The cold-start sequence has the expected shape and is the all-PAD window: every
    # history step is a single one-hot (the PAD slot), so each step sums to 1.0.
    assert seq.shape == (1, pipeline.k, pipeline.step_dim)
    assert np.allclose(seq.sum(axis=2), 1.0)

    probs = fitted_model.predict_proba(x, seq)

    # Valid distribution over exactly the domain option set.
    assert set(probs.keys()) == set(fitted_model.options)
    assert is_valid_distribution(probs)

    # predict() returns an option from the set and confidence is in [0, 1].
    decision = fitted_model.predict(x, seq)
    assert decision in fitted_model.options
    assert 0.0 <= fitted_model.confidence(x, seq) <= 1.0
