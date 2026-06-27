"""Property-based test for feature-pipeline no future leakage.

Property 4: No future leakage
    For any record stream and every feature vector at index ``i``, the ``history`` and
    rolling-frequency components depend only on records strictly earlier than record
    ``i``. Equivalently, transforming the full sequence and transforming any prefix
    ``records[0..p]`` must produce identical features for the shared rows: changing or
    removing records at positions ``>= p`` cannot change the features for records before
    ``p``.

Validates: Requirements 4.3, 4.4
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from data.schema import Domain
from tests.synthetic_data_generator import GeneratorConfig, SyntheticDataGenerator
from features.feature_pipeline import FeaturePipeline
from features.temporal import temporal_width

#: All supported domain values.
_DOMAIN_VALUES = [d.value for d in Domain]


@st.composite
def single_domain_records(draw: st.DrawFn):
    """Draw a small, schema-valid, timestamp-ascending record set for one domain.

    Uses the synthetic generator restricted to a single domain so every record is
    guaranteed schema-valid and the returned list is already sorted ascending by
    timestamp (matching the order the pipeline transforms in). Ranges are kept small
    so the property exercises many record sets quickly.
    """
    domain = draw(st.sampled_from(_DOMAIN_VALUES))
    config = GeneratorConfig(
        n_days=draw(st.integers(min_value=1, max_value=6)),
        decisions_per_day=(1, draw(st.integers(min_value=1, max_value=4))),
        domains=[domain],
        weekend_shift=draw(st.floats(min_value=0.0, max_value=1.0)),
        drift_rate=draw(st.floats(min_value=0.0, max_value=1.0)),
        noise=draw(st.floats(min_value=0.0, max_value=1.0)),
        seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
    )
    records = SyntheticDataGenerator(config).generate()
    k = draw(st.integers(min_value=1, max_value=6))
    return domain, k, records


@given(data=single_domain_records())
@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
def test_prefix_features_match_full_features(data) -> None:
    """Features for records before index ``p`` are independent of records ``>= p``.

    Fit on the full set (learning vocab over all records is allowed -- leakage is
    about the temporal history/rolling components, not the categorical vocabulary),
    transform the full sequence, then transform every prefix and assert the prefix's
    features equal the corresponding leading rows of the full transform.
    """
    domain, k, records = data
    pipeline = FeaturePipeline(domain, k=k).fit(records)

    full = pipeline.transform(records)
    n = full.X.shape[0]

    for p in range(n + 1):
        prefix = pipeline.transform(records[:p])
        assert prefix.X.shape[0] == p
        # Flat feature matrix: prefix rows identical to the full transform's leading p.
        assert np.allclose(prefix.X, full.X[:p]), (
            f"flat features diverged at prefix p={p} (domain={domain}, k={k})"
        )
        # Sequence tensor: same independence guarantee.
        if p > 0:
            assert np.allclose(prefix.seq, full.seq[:p]), (
                f"sequence features diverged at prefix p={p} (domain={domain}, k={k})"
            )


@given(data=single_domain_records())
@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
def test_cold_start_has_no_leakage(data) -> None:
    """The first record sees only PAD history and a uniform rolling-frequency prior.

    At index 0 there are no strictly-earlier records, so the history block must be the
    all-PAD window and the rolling-frequency block must be the uniform prior -- proving
    nothing leaks in from later records.
    """
    domain, k, records = data
    if not records:
        return

    pipeline = FeaturePipeline(domain, k=k).fit(records)
    full = pipeline.transform(records)
    if full.X.shape[0] == 0:
        return

    first = full.X[0]
    n_opts = len(pipeline.options)

    # Layout: [ temporal | rolling-freq | context | last-K history | embedding ].
    rolling_start = temporal_width()
    rolling = first[rolling_start : rolling_start + n_opts]
    assert np.allclose(rolling, np.full(n_opts, 1.0 / n_opts)), (
        "cold-start rolling-frequency block is not the uniform prior"
    )

    history_start = rolling_start + n_opts + pipeline.context_dim
    history = first[history_start : history_start + pipeline.history_dim]
    # All-PAD window: k steps, each a one-hot at the PAD id (slot 0) of vocab_size.
    vocab_size = pipeline.history_dim // k
    expected = np.zeros(pipeline.history_dim, dtype=history.dtype)
    for step in range(k):
        expected[step * vocab_size] = 1.0
    assert np.allclose(history, expected), (
        "cold-start history block is not all-PAD"
    )
