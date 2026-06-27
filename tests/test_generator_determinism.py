"""Property-based test for synthetic data generator determinism.

Property 1: Generator determinism
    For any fixed ``GeneratorConfig`` (including ``seed``), ``generate()`` produces an
    identical record sequence on repeated runs.

Validates: Requirements 1.3
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from data.schema import Domain
from tests.synthetic_data_generator import GeneratorConfig, SyntheticDataGenerator

#: All supported domain values, used to build non-empty subsets.
_DOMAIN_VALUES = [d.value for d in Domain]


@st.composite
def generator_configs(draw: st.DrawFn) -> GeneratorConfig:
    """Draw a varied, valid ``GeneratorConfig``.

    The strategy intentionally constrains the input space to configurations that
    satisfy the generator's preconditions (positive ``n_days``; unit-interval
    ``weekend_shift`` / ``drift_rate`` / ``noise``; a non-empty subset of supported
    domains; a valid ``(min, max)`` decisions-per-day pair). Ranges are kept modest so
    the property exercises many configs quickly.
    """
    lo = draw(st.integers(min_value=0, max_value=5))
    hi = draw(st.integers(min_value=lo, max_value=lo + 5))
    domains = draw(
        st.lists(st.sampled_from(_DOMAIN_VALUES), min_size=1, unique=True)
    )
    return GeneratorConfig(
        n_days=draw(st.integers(min_value=1, max_value=30)),
        decisions_per_day=(lo, hi),
        domains=domains,
        weekend_shift=draw(st.floats(min_value=0.0, max_value=1.0)),
        drift_rate=draw(st.floats(min_value=0.0, max_value=1.0)),
        noise=draw(st.floats(min_value=0.0, max_value=1.0)),
        seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
        user_id=draw(st.sampled_from(["u1", "u2", "alice"])),
    )


@given(config=generator_configs())
@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
def test_generate_is_deterministic_for_fixed_config(config: GeneratorConfig) -> None:
    """Two generators with the same config produce identical record sequences."""
    first = SyntheticDataGenerator(config).generate()
    second = SyntheticDataGenerator(config).generate()

    # Same length and element-wise equality (DecisionRecord is a dataclass, so
    # equality compares every field including the timestamp).
    assert first == second


@given(config=generator_configs())
@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
def test_generate_repeated_on_same_instance_is_deterministic(
    config: GeneratorConfig,
) -> None:
    """Calling generate() twice on the same instance reproduces the sequence."""
    generator = SyntheticDataGenerator(config)
    assert generator.generate() == generator.generate()
