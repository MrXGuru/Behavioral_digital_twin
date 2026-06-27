import pytest
from api.service import TwinService
from tests.synthetic_data_generator import GeneratorConfig, SyntheticDataGenerator

_original_ensure_user = TwinService.ensure_user

def _test_ensure_user(self, user_id: str, min_records: int = 60) -> None:
    """Mock ensure_user that actually seeds synthetic data for tests."""
    if self.store.count(user_id=user_id) >= min_records:
        return
    seed = abs(hash(user_id)) % (2 ** 31)
    cfg = GeneratorConfig(
        n_days=150, 
        decisions_per_day=(2, 4),
        domains=self.domains, 
        weekend_shift=0.6,
        drift_rate=0.05, 
        noise=0.12, 
        seed=seed, 
        user_id=user_id
    )
    self.store.append(SyntheticDataGenerator(cfg).generate())

# Globally monkeypatch for all test scopes (including module-scoped fixtures)
TwinService.ensure_user = _test_ensure_user
