"""Configurable synthetic decision-data generator for the Behavioral Digital Twin.

This module produces a realistic, reproducible dataset of a fictional user's decisions
across the supported domains (``focus`` / ``task`` / ``purchase``). The generated data
exhibits learnable structure -- per-domain habit priors that differ by ``day_type`` and
``time_of_day``, weekday/weekend differences, gradual day-to-day drift -- while remaining
imperfect via bounded random noise.

The generator is deterministic for a fixed :class:`GeneratorConfig` (including ``seed``)
so datasets are fully reproducible, and every emitted record satisfies
:func:`data.schema.validate`.

See the "Synthetic Data Generation" pseudocode in the design document for the contract:

    Preconditions:
      - config.n_days > 0
      - 0 <= config.weekend_shift, config.drift_rate, config.noise <= 1
      - config.domains is a non-empty subset of {focus, task, purchase}
    Postconditions:
      - returns records sorted ascending by timestamp
      - every record passes DecisionRecord validation
      - identical output for identical config.seed (determinism)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from data import schema
from data.schema import (
    DayType,
    DecisionRecord,
    Domain,
    TimeOfDay,
    day_type as derive_day_type,
    options as domain_options,
    time_of_day as derive_time_of_day,
)

# ---------------------------------------------------------------------------
# Generation constants
# ---------------------------------------------------------------------------

#: Fixed calendar anchor so output depends only on the config (and thus the seed).
START_DATE: datetime = datetime(2024, 1, 1)

#: Categorical context vocabularies used when sampling a decision's situation.
LOCATIONS: tuple[str, ...] = ("home", "work", "gym", "store", "transit")
WEATHER: tuple[str, ...] = ("clear", "cloudy", "rain", "snow")

#: Realized-outcome label space (domain-agnostic satisfaction proxy).
OUTCOMES: tuple[str, ...] = ("good", "neutral", "bad")

#: Dirichlet concentration for initial habit priors. Values < 1 yield peaked
#: (i.e. habitual, learnable) distributions over each domain's option set.
_PRIOR_CONCENTRATION: float = 0.5

#: Standard deviation of the additive perturbation applied on a drift day.
_DRIFT_MAGNITUDE: float = 0.08

#: Mean mood/energy by time-of-day, giving ``mood_energy`` learnable structure.
_MOOD_MEAN_BY_TOD: dict[TimeOfDay, float] = {
    TimeOfDay.MORNING: 0.7,
    TimeOfDay.AFTERNOON: 0.55,
    TimeOfDay.EVENING: 0.45,
    TimeOfDay.NIGHT: 0.3,
}
_MOOD_STD: float = 0.15


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GeneratorConfig:
    """Configuration controlling synthetic data generation.

    Attributes:
        n_days: Number of consecutive calendar days to generate (must be > 0).
        decisions_per_day: Inclusive ``(min, max)`` range of decision events per day.
            Each event emits one record per configured domain.
        domains: Non-empty subset of the supported domain values
            (``"focus"`` / ``"task"`` / ``"purchase"``).
        weekend_shift: Strength in ``[0, 1]`` of how different weekend habits are from
            weekday habits (0 = identical, 1 = fully distinct weekend behavior).
        drift_rate: Per-day probability in ``[0, 1]`` that habit priors gradually drift.
        noise: Probability in ``[0, 1]`` that a given decision is a uniformly random
            choice rather than drawn from the habit prior.
        seed: Seed for the random number generator; fixes the entire output.
        user_id: Identifier stamped on every generated record.

    Raises:
        ValueError: if any precondition is violated (descriptive message).
    """

    n_days: int
    decisions_per_day: tuple[int, int]
    domains: list[str]
    weekend_shift: float
    drift_rate: float
    noise: float
    seed: int
    user_id: str = "u1"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Enforce all configuration preconditions, raising ``ValueError`` if violated."""
        # n_days > 0
        if not isinstance(self.n_days, int) or isinstance(self.n_days, bool):
            raise ValueError(
                f"n_days must be an int, got {type(self.n_days).__name__}"
            )
        if self.n_days <= 0:
            raise ValueError(f"n_days must be greater than 0, got {self.n_days}")

        # decisions_per_day: a (min, max) pair with 0 <= min <= max.
        dpd = self.decisions_per_day
        if (
            not isinstance(dpd, (tuple, list))
            or len(dpd) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in dpd)
        ):
            raise ValueError(
                "decisions_per_day must be a (min, max) pair of ints, got "
                f"{self.decisions_per_day!r}"
            )
        lo, hi = dpd
        if lo < 0 or hi < lo:
            raise ValueError(
                "decisions_per_day must satisfy 0 <= min <= max, got "
                f"({lo}, {hi})"
            )
        # Normalize to a plain tuple for deterministic downstream use.
        self.decisions_per_day = (int(lo), int(hi))

        # Unit-interval parameters.
        for name in ("weekend_shift", "drift_rate", "noise"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name} must be a number in [0, 1], got "
                    f"{type(value).__name__}"
                )
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{name} must be within [0, 1] inclusive, got {value}"
                )

        # domains: non-empty subset of the supported domains.
        if not self.domains:
            raise ValueError("domains must be a non-empty list")
        supported = {d.value for d in Domain}
        invalid = [d for d in self.domains if d not in supported]
        if invalid:
            valid = ", ".join(sorted(supported))
            raise ValueError(
                f"domains contains unsupported value(s) {invalid!r}; "
                f"valid domains are: {valid}"
            )

        # seed must be a valid RNG seed.
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an int, got {type(self.seed).__name__}")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

# A habit prior table: domain value -> day_type value -> time_of_day value -> distribution.
_Habits = dict[str, dict[str, dict[str, np.ndarray]]]


@dataclass
class SyntheticDataGenerator:
    """Generate reproducible synthetic :class:`DecisionRecord` sequences.

    The generator builds per-``(domain, day_type, time_of_day)`` habit priors at
    construction-derived state, applies weekday/weekend differences via
    ``weekend_shift``, drifts the priors gradually across days, and injects bounded
    uniform noise. Output is deterministic for a fixed config ``seed``.
    """

    config: GeneratorConfig
    _records: list[DecisionRecord] = field(default_factory=list, init=False)

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self._records = []

    # -- habit prior construction ------------------------------------------

    def _domains(self) -> list[Domain]:
        """Return configured domains as :class:`Domain` enums, in config order."""
        return [Domain(d) for d in self.config.domains]

    def _init_habit_priors(self, rng: np.random.Generator) -> _Habits:
        """Build initial habit priors per ``(domain, day_type, time_of_day)``.

        Weekday priors are drawn as peaked Dirichlet distributions. Weekend priors are a
        ``weekend_shift``-weighted blend between the weekday prior and an independent
        weekend-base draw, so the two day types differ in proportion to ``weekend_shift``.
        """
        shift = float(self.config.weekend_shift)
        habits: _Habits = {}
        for domain in self._domains():
            k = len(domain_options(domain))
            alpha = np.full(k, _PRIOR_CONCENTRATION)
            habits[domain.value] = {dt.value: {} for dt in DayType}
            for tod in TimeOfDay:
                weekday_dist = rng.dirichlet(alpha)
                weekend_base = rng.dirichlet(alpha)
                weekend_dist = (1.0 - shift) * weekday_dist + shift * weekend_base
                # Convex combination of two simplex points stays on the simplex,
                # but renormalize defensively against floating-point drift.
                weekend_dist = weekend_dist / weekend_dist.sum()
                habits[domain.value][DayType.WEEKDAY.value][tod.value] = weekday_dist
                habits[domain.value][DayType.WEEKEND.value][tod.value] = weekend_dist
        return habits

    def _apply_drift(self, habits: _Habits, rng: np.random.Generator) -> _Habits:
        """Gradually perturb every habit distribution for one day.

        With per-day probability ``drift_rate`` a small Gaussian perturbation is added to
        each distribution, then clipped to non-negative and renormalized. This produces
        slow, cumulative concept drift across the simulated horizon.
        """
        if rng.random() >= float(self.config.drift_rate):
            return habits
        for day_map in habits.values():
            for tod_map in day_map.values():
                for key, dist in tod_map.items():
                    perturbed = dist + rng.normal(0.0, _DRIFT_MAGNITUDE, size=dist.shape)
                    perturbed = np.clip(perturbed, 0.0, None)
                    total = perturbed.sum()
                    if total <= 0.0:
                        # Degenerate perturbation: fall back to uniform.
                        perturbed = np.full(dist.shape, 1.0 / dist.size)
                    else:
                        perturbed = perturbed / total
                    tod_map[key] = perturbed
        return habits

    # -- per-record sampling -----------------------------------------------

    def _sample_timestamp(self, day_index: int, rng: np.random.Generator) -> datetime:
        """Sample a within-day timestamp on the given day offset from ``START_DATE``."""
        base = START_DATE + timedelta(days=day_index)
        hour = int(rng.integers(0, 24))
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        return base.replace(hour=hour, minute=minute, second=second, microsecond=0)

    def _sample_mood(self, tod: TimeOfDay, rng: np.random.Generator) -> float:
        """Sample a ``mood_energy`` value in ``[0, 1]`` with time-of-day structure."""
        mean = _MOOD_MEAN_BY_TOD[tod]
        value = float(rng.normal(mean, _MOOD_STD))
        return float(np.clip(value, 0.0, 1.0))

    def _sample_context(
        self, rng: np.random.Generator
    ) -> tuple[str, str]:
        """Sample categorical context (``location``, ``weather``)."""
        location = LOCATIONS[int(rng.integers(0, len(LOCATIONS)))]
        weather = WEATHER[int(rng.integers(0, len(WEATHER)))]
        return location, weather

    def _sample_choice(
        self,
        habits: _Habits,
        domain: Domain,
        day_type_value: str,
        tod: TimeOfDay,
        rng: np.random.Generator,
    ) -> str:
        """Choose a decision: bounded uniform noise, else draw from the habit prior."""
        opts = domain_options(domain)
        if rng.random() < float(self.config.noise):
            return opts[int(rng.integers(0, len(opts)))]
        dist = habits[domain.value][day_type_value][tod.value]
        idx = int(rng.choice(len(opts), p=dist))
        return opts[idx]

    def _realize_outcome(
        self, choice: str, mood: float, rng: np.random.Generator
    ) -> str:
        """Realize an outcome label, biased by ``mood_energy``."""
        # Higher mood -> more likely a "good" outcome.
        good = 0.2 + 0.6 * mood
        bad = 0.2 + 0.6 * (1.0 - mood)
        neutral = max(0.0, 1.0 - good - bad)
        probs = np.array([good, neutral, bad])
        probs = probs / probs.sum()
        idx = int(rng.choice(len(OUTCOMES), p=probs))
        return OUTCOMES[idx]

    # -- public API --------------------------------------------------------

    def generate(self) -> list[DecisionRecord]:
        """Generate the full record sequence, sorted ascending by timestamp.

        Returns:
            A list of :class:`DecisionRecord`, every element of which passes
            :func:`data.schema.validate`, sorted by ``timestamp`` ascending.
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        habits = self._init_habit_priors(rng)
        lo, hi = cfg.decisions_per_day

        records: list[DecisionRecord] = []
        for day in range(cfg.n_days):
            # Drift is evaluated once per day so priors evolve gradually over time.
            habits = self._apply_drift(habits, rng)
            n_events = int(rng.integers(lo, hi + 1))
            for _ in range(n_events):
                ts = self._sample_timestamp(day, rng)
                tod = derive_time_of_day(ts)
                dt = derive_day_type(ts)
                mood = self._sample_mood(tod, rng)
                for domain in self._domains():
                    location, weather = self._sample_context(rng)
                    choice = self._sample_choice(habits, domain, dt.value, tod, rng)
                    outcome = self._realize_outcome(choice, mood, rng)
                    record = DecisionRecord(stress_level="low", 
                        user_id=cfg.user_id,
                        timestamp=ts,
                        domain=domain.value,
                        location=location,
                        weather=weather,
                        day_type=dt.value,
                        time_of_day=tod.value,
                        mood_energy=mood,
                        decision_made=choice,
                        outcome=outcome,
                    )
                    schema.validate(record)
                    records.append(record)

        records.sort(key=lambda r: r.timestamp)
        self._records = records
        return records

    def to_dataframe(self) -> "pd.DataFrame":  # noqa: F821 (lazy import)
        """Return the generated records as a pandas DataFrame in schema column order.

        Generates the records first if :meth:`generate` has not yet been called.
        """
        import pandas as pd

        records = self._records if self._records else self.generate()
        rows = [
            {col: getattr(record, col) for col in schema.COLUMNS} for record in records
        ]
        return pd.DataFrame(rows, columns=list(schema.COLUMNS))
