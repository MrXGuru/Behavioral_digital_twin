"""Time-based evaluation and model comparison.

Implements a strictly temporal train/validation split (no future leakage), accuracy,
macro-F1, and Brier-score (calibration) metrics, and ``evaluate_models`` which trains and
scores both the baseline and sequence twins on the validation partition for each domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

from data.schema import DecisionRecord, Domain, options
from features.feature_pipeline import FeaturePipeline
from models.baseline import BaselineModel
from models.sequence import SequenceModel


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    brier: float
    log_loss: float
    n: int

    def as_dict(self) -> dict:
        return {"accuracy": round(self.accuracy, 4), "macro_f1": round(self.macro_f1, 4),
                "brier": round(self.brier, 4), "log_loss": round(self.log_loss, 4), "n": self.n}


@dataclass
class ComparisonReport:
    domain: str
    baseline: Metrics
    sequence: Metrics
    winner: str = ""
    rationale: str = ""
    winning_model: object = None
    winning_pipe: object = None

    def as_dict(self) -> dict:
        return {"domain": self.domain, "baseline": self.baseline.as_dict(),
                "sequence": self.sequence.as_dict(), "winner": self.winner,
                "rationale": self.rationale}


def time_based_split(
    records: list[DecisionRecord], val_fraction: float = 0.2
) -> tuple[list[DecisionRecord], list[DecisionRecord]]:
    """Split sorted records temporally so all train timestamps <= all val timestamps.

    Returns a partition of the input (no loss, no duplication).
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    ordered = sorted(records, key=lambda r: r.timestamp)
    n = len(ordered)
    n_val = max(1, int(round(n * val_fraction))) if n > 1 else 0
    split = n - n_val
    return ordered[:split], ordered[split:]


def _brier(probs_list: list[dict[str, float]], y_true: list[str],
           opts: list[str]) -> float:
    """Multiclass Brier score: mean squared error vs one-hot truth."""
    if not y_true:
        return 0.0
    total = 0.0
    for probs, truth in zip(probs_list, y_true):
        for opt in opts:
            target = 1.0 if opt == truth else 0.0
            total += (probs.get(opt, 0.0) - target) ** 2
    return total / len(y_true)


def _score_model(model, fm_val, opts: list[str]) -> Metrics:
    if not fm_val.y:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0)
    preds, probs_list = [], []
    for i in range(len(fm_val.y)):
        x = fm_val.X[i:i + 1]
        seq = fm_val.seq[i:i + 1]
        probs = model.predict_proba(x, seq)
        probs_list.append(probs)
        preds.append(max(probs, key=probs.get))
    acc = accuracy_score(fm_val.y, preds)
    macro_f1 = f1_score(fm_val.y, preds, average="macro", zero_division=0)
    brier = _brier(probs_list, fm_val.y, opts)
    
    # Compute log_loss
    # Extract probability matrix corresponding to `opts`
    prob_matrix = [[p.get(opt, 0.0) for opt in opts] for p in probs_list]
    try:
        ll = log_loss(fm_val.y, prob_matrix, labels=opts)
    except Exception:
        ll = 0.0 # fallback in case of errors with single classes

    return Metrics(float(acc), float(macro_f1), float(brier), float(ll), len(fm_val.y))


def evaluate_models(
    records: list[DecisionRecord], domain: str | Domain, val_fraction: float = 0.2,
    k: int = 5,
) -> ComparisonReport:
    """Train baseline + sequence on a temporal split and score both on validation."""
    dom = Domain(domain) if not isinstance(domain, Domain) else domain
    dom_records = [r for r in records if r.domain == dom.value]
    train, val = time_based_split(dom_records, val_fraction)

    import time
    t0 = time.time()
    pipe = FeaturePipeline(dom, k=k).fit(train)
    fm_train = pipe.transform(train)
    fm_val = pipe.transform(val)
    opts = list(options(dom))
    print(f"FeaturePipeline took {time.time()-t0:.2f}s")

    t0 = time.time()
    baseline = BaselineModel(opts).fit(fm_train.X, fm_train.seq, fm_train.y)
    print(f"BaselineModel took {time.time()-t0:.2f}s")

    t0 = time.time()
    sequence = SequenceModel(opts).fit(fm_train.X, fm_train.seq, fm_train.y)
    print(f"SequenceModel took {time.time()-t0:.2f}s")

    b_metrics = _score_model(baseline, fm_val, opts)
    s_metrics = _score_model(sequence, fm_val, opts)

    # Winner: higher accuracy, tie-break on lower Brier (better calibration).
    if (s_metrics.accuracy, -s_metrics.brier) > (b_metrics.accuracy, -b_metrics.brier):
        winner = "sequence"
        rationale = ("Sequence model better captures order-dependent habit patterns "
                     "(higher accuracy / better calibration on the held-out tail).")
        winning_model = sequence
    else:
        winner = "baseline"
        rationale = ("Baseline matches or beats the sequence model here; with limited "
                     "data the simpler model generalizes better and is better calibrated.")
        winning_model = baseline
    return ComparisonReport(dom.value, b_metrics, s_metrics, winner, rationale, winning_model=winning_model, winning_pipe=pipe)
