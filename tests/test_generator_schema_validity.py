"""Property-based tests for schema validity of generated decision records.

Property 2: Schema validity
    *For any* generated or stored ``DecisionRecord``, all ``DecisionRecord`` validation
    rules hold: ``domain`` is valid, ``decision_made`` belongs to the domain option set,
    ``mood_energy`` is in [0, 1], and ``time_of_day`` / ``day_type`` are consistently
    derived from the timestamp.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from data import schema
from data.schema import DOMAIN_OPTIONS, Domain, day_type, time_of_day
from tests.synthetic_data_generator import GeneratorConfig, SyntheticDataGenerator

# All supported domain string values.
_SUPPORTED_DOMAINS = [d.value for d in Domain]


@st.composite
def generator_configs(draw: st.DrawFn) -> GeneratorConfig:
    """Draw a varied but always-valid ``GeneratorConfig``.

    The strategy stays within the generator's precondition space (n_days > 0;
    weekend_shift/drift_rate/noise in [0, 1]; domains a non-empty subset of the
    supported domains; valid decisions_per_day range) so every config is acceptable
    and exercises the full breadth of generation behavior.
    """
    n_days = draw(st.integers(min_value=1, max_value=12))

    lo = draw(st.integers(min_value=0, max_value=4))
    hi = draw(st.integers(min_value=lo, max_value=lo + 4))

    domains = draw(
        st.lists(
            st.sampled_from(_SUPPORTED_DOMAINS),
            min_size=1,
            max_size=len(_SUPPORTED_DOMAINS),
            unique=True,
        )
    )

    unit = st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    )
    weekend_shift = draw(unit)
    drift_rate = draw(unit)
    noise = draw(unit)

    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    user_id = draw(st.text(min_size=1, max_size=8))

    return GeneratorConfig(
        n_days=n_days,
        decisions_per_day=(lo, hi),
        domains=domains,
        weekend_shift=weekend_shift,
        drift_rate=drift_rate,
        noise=noise,
        seed=seed,
        user_id=user_id,
    )


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(config=generator_configs())
def test_generated_records_satisfy_schema(config: GeneratorConfig) -> None:
    """Every generated record passes full schema validation (Property 2)."""
    records = SyntheticDataGenerator(config).generate()

    for record in records:
        # validate() raises SchemaValidationError on any violated rule; if it
        # returns, all rules hold for this record.
        schema.validate(record)
        assert schema.is_valid(record)


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(config=generator_configs())
def test_generated_records_satisfy_each_rule(config: GeneratorConfig) -> None:
    """Spell out each individual rule from Property 2 across all generated records."""
    records = SyntheticDataGenerator(config).generate()

    for record in records:
        # Rule 1: domain is one of the supported domains.
        assert record.domain in _SUPPORTED_DOMAINS

        # Rule 2: decision_made belongs to the domain's option set.
        assert record.decision_made in DOMAIN_OPTIONS[Domain(record.domain)]

        # Rule 3: mood_energy in [0, 1].
        assert 0.0 <= record.mood_energy <= 1.0

        # Rule 4: time_of_day consistent with the timestamp hour.
        assert record.time_of_day == time_of_day(record.timestamp).value

        # Rule 5: day_type consistent with the timestamp weekday.
        assert record.day_type == day_type(record.timestamp).value
