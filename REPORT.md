# Behavioral Digital Twin — Evaluation and Ethics Report

This report documents how the two competing twin models compare across domains, the
system's privacy posture, the misuse risks we considered and how they are mitigated, and
the explicit security flag covering the unauthenticated demo deployment.

All numbers below are **measured on synthetic data** produced by the project's own
generator (`data/synthetic_data_generator.py`). They are reproducible from the config
stated under each table.

---

## 1. Model Comparison Results (Req 12.2)

We compare two model families on every supported domain:

- **BaselineModel** — a context-conditioned frequency/logistic baseline.
- **SequenceModel** — an order-aware model that uses the recent decision history.

### Methodology

- Evaluation uses a **strictly temporal** train/validation split
  (`models/evaluate.py:time_based_split`, `val_fraction=0.2`): every training timestamp is
  `<=` every validation timestamp, so there is no future leakage.
- Metrics per domain:
  - **Accuracy** — fraction of validation decisions predicted exactly.
  - **macro-F1** — unweighted mean F1 across the domain's option classes (so rare options
    are not drowned out by common ones).
  - **Brier_Score** — multiclass Brier score (mean squared error of the predicted
    probability vector versus the one-hot truth); **lower is better**, it measures
    calibration.
- **Winner rule** (`evaluate_models`): higher accuracy wins, ties broken by lower (better)
  Brier score.

### Dataset / config used

Measured on a single synthetic user generated with:

```python
GeneratorConfig(
    n_days=120,
    decisions_per_day=(3, 6),
    domains=["focus", "task", "purchase"],
    weekend_shift=0.4,
    drift_rate=0.1,
    noise=0.1,
    seed=42,
    user_id="u1",
)
```

This produced **1653 total records**; each domain's validation tail held **n = 110**
decisions. Because the generator is deterministic for a fixed seed, these results are
fully reproducible.

### Results

**Domain: `focus`** (options: pomodoro / flow_state / light_work / admin)

| Model         | Accuracy | macro-F1 | Brier_Score | n   |
| ------------- | -------- | -------- | ----------- | --- |
| BaselineModel | 0.3727   | 0.3281   | 0.8803      | 110 |
| SequenceModel | 0.4455   | 0.4302   | 0.7702      | 110 |

**Winner: SequenceModel.** It wins on all three metrics — higher accuracy, higher
macro-F1, and better (lower) Brier calibration — indicating focus choice carries
order-dependent habit structure the sequence model captures.

**Domain: `task`** (options: deep_work / email / meeting / break)

| Model         | Accuracy | macro-F1 | Brier_Score | n   |
| ------------- | -------- | -------- | ----------- | --- |
| BaselineModel | 0.4818   | 0.3309   | 0.8678      | 110 |
| SequenceModel | 0.5091   | 0.3832   | 0.6916      | 110 |

**Winner: SequenceModel.** Accuracy is close, but the sequence model is meaningfully
better calibrated (Brier 0.69 vs 0.87) and has stronger macro-F1, so it handles the
less-frequent task types better.

**Domain: `purchase`** (options: coffee / snack / lunch / none)

| Model         | Accuracy | macro-F1 | Brier_Score | n   |
| ------------- | -------- | -------- | ----------- | --- |
| BaselineModel | 0.3364   | 0.2573   | 0.9734      | 110 |
| SequenceModel | 0.3636   | 0.3417   | 0.8232      | 110 |

**Winner: SequenceModel.** Purchase is the hardest domain (most noise relative to
signal), but the sequence model still leads on every metric, most notably calibration
(Brier 0.82 vs 0.97) and macro-F1.

### Summary

The **SequenceModel wins in all three domains** on this dataset, with its largest, most
consistent advantage being **calibration (Brier)**. The intuition: these decisions are
habitual and order-dependent (what you did recently predicts what you do next), which a
sequence-aware model exploits and a context-only baseline cannot. Absolute accuracy stays
modest by design — the generator injects bounded noise and gradual drift, so the data is
deliberately imperfect rather than trivially separable.

> Reproduce: generate a dataset with the config above and call
> `models.evaluate.evaluate_models(records, domain)` for each domain.

---

## 2. Privacy Posture (Req 12.3)

This system models **private human behavior**, so privacy is treated as a first-class
design constraint.

- **Synthetic-by-default.** All data is synthetic unless a real source is explicitly
  enabled. The default pipeline runs entirely on
  `data/synthetic_data_generator.py` output, every record carries `synthetic`/`real`
  provenance (`schema.SOURCE_MODES`), and no real personal data is required to run,
  demo, train, or evaluate the system.
- **Data minimization.** The schema (`data/schema.py:DecisionRecord`) captures only the
  coarse fields needed to learn habits: a user id, timestamp, domain, a few low-resolution
  context categories (location, weather, day_type, time_of_day), a `[0,1]` mood/energy
  proxy, the chosen option, and an outcome label. No free-text, no precise geolocation, no
  device identifiers, and no sensitive-category attributes are collected.
- **User consent.** Real data may only enter the system through an explicit, opt-in source
  mode; the synthetic default means the system never ingests a person's real decisions
  without a deliberate choice to enable a real connector.
- **Per-user isolation.** Records are keyed by `user_id` and the persistence boundary
  (`data/decision_store.py:DecisionStore`) filters every `load`/`count` by user.
  Personalization (`personalization/`) builds and stores **per-user** profiles, and the
  API addresses every operation by `{user_id}`, so one user's data and twin never blend
  into another's.
- **User-controlled delete/export path.** Because all of a user's data is reachable
  through the single `DecisionStore` boundary keyed by `user_id`:
  - **Export** is supported by loading that user's records
    (`DecisionStore.load(user_id=...)`) and serializing them (CSV/SQLite round-trip in the
    canonical schema column order).
  - **Delete** is supported by removing that user's records from the backing store.
  - These give the user a right-to-access and right-to-erasure path. **Note:** the delete
    endpoint is not yet surfaced in the demo API surface (`api/main.py`) and must be wired
    up (and protected by authorization — see §4/§5) before any non-demo exposure.

---

## 3. Misuse Mitigations (Req 12.4)

A model that predicts personal decisions can be abused. We name the risks explicitly and
state the mitigations.

- **Manipulation risk.** A behavior predictor could be used to nudge or exploit a person
  (e.g., timing offers to override their intent).
  - *Mitigation:* the twin is positioned as an **explain-only / decision-support** tool,
    not an actuator. The chat surface is explain-only and is constrained to the model's
    actual structured prediction (it cannot invent a different predicted decision or
    confidence). Outputs are probabilities surfaced to the user about their own behavior,
    not levers exposed to third parties.
- **Surveillance risk.** Continuous behavioral logging can become covert tracking.
  - *Mitigation:* synthetic-by-default operation, strict data minimization (coarse
    categorical context only — no precise location, no raw text), and per-user isolation
    limit how much can be learned and ensure data is scoped to its owner. Real data
    requires explicit opt-in.
- **Prediction transparency.** Opaque predictions are easier to misuse and harder to
  contest.
  - *Mitigation:* every prediction is returned with a **calibrated probability
    distribution** (`class_probs`) and a `confidence` equal to `max(class_probs)`, plus a
    `drift_status`. The reported Brier scores above document calibration quality, and the
    explain-only chat presents reasoning as prose **on top of** the unchanged structured
    prediction.
- **User control.** The subject of the prediction should be in charge.
  - *Mitigation:* the user can inspect their own decisions, profile, and predictions; can
    export and delete their data (§2); and the personalization layer adapts to *their*
    recent decisions rather than imposing a population model.
- **Opt-out.** Participation must be revocable.
  - *Mitigation:* because the system is synthetic-by-default and real data is opt-in per
    user, a user can opt out by not enabling (or disabling) the real source and by
    deleting their stored records via the delete path. Opt-out leaves the rest of the
    system fully functional on synthetic data.

---

## 4. Deployment / Authentication Flag (Req 12.5)

**⚠️ The demo API is deployed WITHOUT authentication.**

`api/main.py` exposes the prediction, history, drift, retrain, profile, and full-dashboard
endpoints with **no authentication and no per-user authorization**, and with permissive
(`*`) CORS. Any caller can request **any** `{user_id}`. This is acceptable **for a local,
single-user demo only**.

**Required before any non-demo exposure:**

1. **Authentication** — require verified identity on every endpoint (no anonymous access).
2. **Per-user authorization** — enforce that an authenticated caller may only read or
   modify their **own** `{user_id}` data; reject cross-user access.
3. **Lock down CORS** to known origins instead of `*`.
4. **Protect the delete/export path** (§2) behind the same authentication and
   authorization so erasure/export can only be triggered by the data's owner.

Because this system models **private human behavior**, deploying it as-is to any shared or
public environment would expose every user's behavioral data to every caller. Treat the
no-auth configuration as a demo-only constraint that **must** be remediated first.
