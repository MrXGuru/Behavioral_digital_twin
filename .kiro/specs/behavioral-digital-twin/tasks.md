# Implementation Plan: Behavioral Digital Twin

## Overview

This plan implements the Behavioral Digital Twin in Python as a fully demoable
repository, built incrementally in the order: data generator and persistence → feature
engineering → baseline model and evaluation → sequence model → FastAPI prediction engine
→ personalization and concept-drift adaptation → end-to-end demos → Streamlit dashboard →
README/REPORT.

Each task builds on the prior ones and ends by wiring new code into the running system,
so nothing is left orphaned. The FastAPI service is first stood up serving model
predictions, then enhanced with per-user profile features and drift status, matching the
requested "API → personalization/drift" order while keeping every step buildable.

Testing is interleaved with implementation. The 8 correctness properties from the design
are turned into `hypothesis` property-based tests placed next to the code they validate;
`pytest` unit and integration tests cover edge cases and end-to-end flows. Tasks marked
with `*` are optional test sub-tasks and can be skipped for a faster MVP.

## Tasks

- [x] 1. Set up project structure and shared schema
  - [x] 1.1 Create repository skeleton and packaging
    - Create package directories `data/`, `features/`, `models/`, `personalization/`, `api/`, `demos/`, `dashboard/`, `notebooks/`, `tests/`, each with `__init__.py`
    - Create `requirements.txt` (numpy, pandas, scikit-learn, torch, fastapi, uvicorn, pydantic, streamlit, plotly, pytest, hypothesis, httpx) and `pyproject.toml`/pytest config
    - Create `.gitignore` excluding `data/generated/`, `models/artifacts/`, `__pycache__/`, `*.db`, virtualenv folders, with `.gitkeep` placeholders in `data/generated/` and `models/artifacts/`
    - _Requirements: 1.1_

  - [x] 1.2 Implement `data/schema.py` with the DecisionRecord schema, enums, and validation
    - Define `Domain` enum and per-domain option sets for `route`, `task`, `purchase`; define `DayType`/`TimeOfDay` enums plus `time_of_day(ts)` and `day_type(ts)` helpers
    - Define the `DecisionRecord` dataclass with all design fields and the canonical `COLUMNS` ordering and `SCHEMA_VERSION`; expose `UNK` and `PAD` constants
    - Implement `validate(record)` enforcing: valid domain, `decision_made` in the domain option set, `mood_energy` in [0,1], and `time_of_day`/`day_type` consistent with the timestamp
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x]* 1.3 Write unit tests for schema validation
    - Test that valid records pass and that each violated rule (bad domain, out-of-range mood, mismatched `time_of_day`/`day_type`, `decision_made` not in option set) is rejected with a descriptive error
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 2. Implement the synthetic data generator and persistence layer
  - [x] 2.1 Implement `data/synthetic_data_generator.py`
    - Define `GeneratorConfig` (n_days, decisions_per_day, domains, weekend_shift, drift_rate, noise, seed) with precondition validation per the design pseudocode
    - Implement `SyntheticDataGenerator` with habit priors per (domain, day_type, time_of_day), weekday/weekend differences, gradual per-day drift, bounded random noise, and a seeded RNG
    - Implement `generate()` returning timestamp-ascending records that all pass `schema.validate`, and `to_dataframe()` conforming to the shared column schema
    - _Requirements: 1.1, 1.2, 1.4, 1.5_

  - [x] 2.2 Write property test for generator determinism
    - **Property 1: Generator determinism**
    - **Validates: Requirements 1.3**

  - [x] 2.3 Write property test for schema validity of generated records
    - **Property 2: Schema validity**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

  - [x] 2.4 Implement `data/decision_store.py` with CSV and SQLite backends
    - Implement `DecisionStore(backend, path)` supporting `backend in {"csv","sqlite"}` behind one interface, reusing the schema column ordering/version so both backends round-trip identically
    - Implement `append(records)`, `load(user_id=None, since=None)` returning records sorted ascending by timestamp, and `count(user_id=None)`
    - _Requirements: 1.2, 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x]* 2.5 Write property test for temporal ordering on load
    - **Property 3: Temporal ordering**
    - **Validates: Requirements 1.2, 3.3**

  - [x]* 2.6 Write unit tests for store round-trip and filtering
    - Test CSV and SQLite append/load round-trip equality, `since` filtering, `user_id` filtering, and `count`
    - _Requirements: 3.1, 3.2, 3.4, 3.5_

- [x] 3. Checkpoint - data layer
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement the feature engineering pipeline
  - [x] 4.1 Implement `features/encoders.py`
    - Implement categorical context encoding fit/transform with a deterministic vocabulary and an explicit `<UNK>` bucket for values unseen during fit, exposing fixed output dimensions
    - _Requirements: 4.5, 4.6_

  - [x] 4.2 Implement `features/temporal.py`
    - Implement hour sin/cos, day-of-week one-hot, and an incremental `RollingFrequencyTracker` (O(1) per record) whose features use only records strictly earlier than the current record
    - _Requirements: 4.1, 4.3_

  - [x] 4.3 Implement `features/history.py`
    - Implement the last-K decision-sequence builder that left-pads with `PAD` to length K and updates the window only after building features for the current record (no future leakage)
    - _Requirements: 4.2, 4.4_

  - [x] 4.4 Implement `features/feature_pipeline.py` orchestration
    - Implement `FeaturePipeline.fit`, `transform` (one `FeatureVector` per record combining temporal + context + history + user embedding), and `transform_one(context, recent, profile)` for the prediction path; expose `k` and `history_dim` and integrate the encoders, temporal, and history modules
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x]* 4.5 Write property test for no future leakage
    - **Property 4: No future leakage**
    - **Validates: Requirements 4.3, 4.4**

  - [x]* 4.6 Write unit tests for encoder UNK handling and fixed dimensions
    - Test unseen categories map to `<UNK>`, fit/transform consistency, and stable sub-vector dimensions
    - _Requirements: 4.5, 4.6_

- [x] 5. Checkpoint - feature pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement the baseline model and evaluation
  - [x] 6.1 Implement `models/base.py` DecisionModel interface
    - Define the abstract `DecisionModel` with `fit`, `predict`, `predict_proba`, `save`, and classmethod `load`; define shared per-domain label-space helpers and a probability-distribution validity helper
    - _Requirements: 5.1, 5.2, 5.5_

  - [x] 6.2 Implement `models/baseline.py`
    - Implement `BaselineModel` (gradient boosting / logistic regression) over flat feature vectors, with `predict_proba` returning a valid distribution, `predict`/`confidence` consistent with `max(class_probs)`, and a save/load round-trip
    - _Requirements: 5.1, 5.3, 5.4, 5.5_

  - [x]* 6.3 Write property test for probability validity
    - **Property 5: Probability validity**
    - **Validates: Requirements 5.3, 5.4, 7.2, 7.3**

  - [x] 6.4 Implement `models/evaluate.py` time-based split, scoring, and comparison
    - Implement `time_based_split(records, val_fraction)` producing a temporal partition (no loss/duplication, all train timestamps ≤ all val timestamps)
    - Implement scoring (accuracy, macro-F1, Brier) and `evaluate_models(records)` returning a `ComparisonReport`, initially scoring the `BaselineModel`
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x]* 6.5 Write property test for temporal split integrity
    - **Property 6: Temporal split integrity**
    - **Validates: Requirements 6.1, 6.2**

  - [x]* 6.6 Write unit tests for baseline save/load round-trip
    - Test that a saved-then-loaded model yields predictions equal to the original for identical inputs
    - _Requirements: 5.5_

- [x] 7. Checkpoint - baseline and evaluation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement the sequence model
  - [x] 8.1 Implement `models/sequence.py`
    - Implement `SequenceModel` using torch (single-layer LSTM or 1–2 block Transformer) consuming the last-K sequence plus context and user embedding, kept small enough to train on CPU; implement `predict_proba`, `predict` (consistent with `max(class_probs)`), and save/load round-trip
    - _Requirements: 5.2, 5.3, 5.4, 5.5_

  - [x]* 8.2 Write unit tests for sequence model save/load and cold-start padding
    - Test predictions on padded (cold-start) histories and artifact round-trip equality
    - _Requirements: 5.5_

  - [x] 8.3 Wire the sequence model into model comparison
    - Extend `evaluate_models` so the `ComparisonReport` includes accuracy, macro-F1, and Brier for both the baseline and the sequence model on the validation partition
    - _Requirements: 6.3, 6.4_

- [x] 9. Implement the FastAPI prediction engine
  - [x] 9.1 Implement `api/schemas.py` pydantic models
    - Define request/response models for `/predict_next_decision`, `/user_profile/{id}`, and `/retrain`, mirroring `Context`, `PredictionResult`, and `DriftStatus`
    - _Requirements: 7.1, 7.4, 7.5_

  - [x] 9.2 Implement `api/main.py` app and endpoints (model-serving core)
    - Wire the feature pipeline and models into `POST /predict_next_decision` (returning `predicted_decision`, `confidence`, `class_probs`), `GET /user_profile/{id}`, and `POST /retrain` (returning status + evaluation metrics)
    - Implement error handling: map unseen categories to `<UNK>`, left-pad cold-start histories, return HTTP 200 `{status: "skipped", reason: "insufficient_data"}` on too-few-records retrain (keep serving prior artifact), and HTTP 409 directing to `/retrain` when a domain's model artifact is missing
    - _Requirements: 7.1, 7.3, 7.5, 7.6, 7.7_

  - [x]* 9.3 Write integration tests for the API via TestClient
    - Test predict returns a valid response shape, retrain happy-path metrics, insufficient-data skip (200), and missing-artifact 409
    - _Requirements: 7.1, 7.3, 7.5, 7.6, 7.7_

- [x] 10. Implement personalization and concept-drift adaptation
  - [x] 10.1 Implement `personalization/profile_store.py`
    - Define `UserProfile` (user_id, embedding, decision_counts, last_updated) and `UserProfileStore` with `get` (returning a cold-start default for unknown users) and `upsert`
    - _Requirements: 11.1_

  - [x] 10.2 Implement `personalization/updater.py`
    - Implement `ProfileUpdater.update` incrementing each decision count by new-record occurrences, moving the embedding toward aggregate new-record behavior via an EMA update, and setting `last_updated` to the max timestamp among the new records
    - _Requirements: 11.2, 11.3, 11.4_

  - [x]* 10.3 Write property test for profile monotonicity
    - **Property 7: Profile monotonicity**
    - **Validates: Requirements 11.2, 11.4**

  - [x] 10.4 Implement `api/drift.py` DriftDetector
    - Implement `DriftDetector(window, threshold)` with `record(predicted, actual, confidence)` updating a rolling record, and `status()` returning `DriftStatus` where `drift` is True iff at least `window` labeled predictions exist and rolling accuracy < threshold (and `window_acc` is None with no observations)
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x]* 10.5 Write property test for drift flag soundness
    - **Property 8: Drift flag soundness**
    - **Validates: Requirements 10.3, 10.4**

  - [x] 10.6 Integrate profile features and drift status into the API (wiring)
    - Feed the per-user profile embedding from `UserProfileStore` into the feature build for `/predict_next_decision`, record each prediction with the `DriftDetector`, and include `drift_status` in the response
    - Implement `GET /user_profile/{id}` to return decision counts, embedding summary, and last-updated from the profile store, and refresh the user's profile embedding via `ProfileUpdater` inside `/retrain`
    - _Requirements: 7.1, 7.2, 7.4, 11.1_

  - [x]* 10.7 Write integration tests for personalization and drift
    - Test `drift_status` appears in the predict response, the `/user_profile/{id}` shape, and profile refresh on retrain
    - _Requirements: 7.1, 7.4, 10.1, 11.1_

- [x] 11. Checkpoint - serving and adaptation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Implement end-to-end demos
  - [x] 12.1 Implement `demos/route_demo.py`
    - Run the full generate → store → feature build → train → predict path for a route-choice prediction with confidence, without manual intervention
    - _Requirements: 8.1, 8.4_

  - [x] 12.2 Implement `demos/productivity_demo.py`
    - End-to-end productivity task-choice demo following the same full path
    - _Requirements: 8.2, 8.4_

  - [x] 12.3 Implement `demos/purchase_demo.py`
    - End-to-end purchase-choice demo following the same full path
    - _Requirements: 8.3, 8.4_

  - [x]* 12.4 Write demo smoke tests
    - Test each demo runs end-to-end and produces a valid prediction without error
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 13. Implement the Streamlit dashboard
  - [x] 13.1 Implement `dashboard/app.py`
    - Build the decision timeline (ascending by timestamp), prediction-vs-actual view, confidence-over-time chart, and drift alerts, reading from the store/API and drift detector; factor chart inputs through pure data-prep helpers
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x]* 13.2 Write unit tests for dashboard data-prep helpers
    - Test the pure data-shaping functions feeding the charts (timeline ordering, confidence series, drift status mapping)
    - _Requirements: 9.1, 9.2, 9.3_

- [x] 14. Documentation and final wiring
  - [x] 14.1 Write `README.md`
    - Document setup and how to generate data, train, serve the API via uvicorn, run the three demos, and launch the dashboard, with an architecture overview
    - _Requirements: 12.1_

  - [x] 14.2 Write `REPORT.md`
    - Document model comparison results (Accuracy, macro-F1, Brier), the privacy posture (synthetic-by-default, data minimization, consent, per-user isolation, delete/export), misuse mitigations (manipulation, surveillance, transparency, user control, opt-out), and the explicit no-auth demo flag
    - _Requirements: 12.2, 12.3, 12.4, 12.5_

- [x] 15. Checkpoint - original core suite
  - Ensure all tests pass for the original core (data, features, models, API, personalization, drift, demos, docs), ask the user if questions arise.

- [x] 16. Implement the dual-mode data layer (Phase 1 extension)
  - [x] 16.1 Add `source_mode` to the schema and persistence
    - Add the `source_mode` field to `DecisionRecord` and `COLUMNS` in `data/schema.py` (values `synthetic`/`real`, default `synthetic`), bump `SCHEMA_VERSION`, and extend `validate` to enforce the allowed values; update `DecisionStore` CSV/SQLite serialization so both backends round-trip the new column and existing rows without `source_mode` default to `synthetic`
    - _Requirements: 13.2_

  - [x] 16.2 Implement `data/sources/base.py` (mode-agnostic contract)
    - Define the `SourceMode` enum and the abstract `DataSource` interface (`fetch(user_id, since=None)`, `count(user_id)`) returning schema-valid, timestamp-ascending records and a non-negative count
    - _Requirements: 13.1, 13.7_

  - [x] 16.3 Implement `data/sources/synthetic_source.py`
    - Implement `SyntheticDataSource` wrapping the existing `SyntheticDataGenerator`, stamping every emitted record `source_mode="synthetic"`, satisfying the `DataSource` contract unchanged
    - _Requirements: 13.1, 13.2_

  - [x] 16.4 Implement connector interface and concrete connectors
    - Implement `data/sources/connectors/base_connector.py` (`SourceConnector` with `name`, `enabled`, `pull(user_id, since=None)`); implement concrete `UserInteractionsConnector`, `RouteSelectionsConnector`, and `CsvImportConnector` returning schema-valid records tagged `real`
    - _Requirements: 13.5_

  - [x] 16.5 Implement stub connectors behind the same contract
    - Implement `data/sources/connectors/stubs.py` with `StudySessionsConnector`, `ProductivityLogsConnector`, `CalendarConnector`, `GitHubActivityConnector`, `BrowserActivityConnector` (opt-in only), `WeatherConnector`, `TimeConnector`, `DeviceConnector`, each returning an empty but schema-valid result and reporting `enabled = False`
    - _Requirements: 13.5_

  - [x] 16.6 Implement `data/sources/real_source.py`
    - Implement `RealDataSource` merging enabled connectors' output time-ordered, tagging records `source_mode="real"`, and skipping any errored or disabled connector with a structured warning while still returning schema-valid records from the rest
    - _Requirements: 13.1, 13.2, 13.6_

  - [x] 16.7 Implement `data/sources/mode_manager.py`
    - Implement `ModeConfig`, a `ModeConfigStore`, and `ModeManager` with `get_mode` (defaulting new users to synthetic), `resolve_source` (returns the active `DataSource`), and `evaluate_migration` (flip synthetic→real at `migration_threshold`, keep mode non-decreasing, ramp `blend_weight` in [0,1] non-decreasing)
    - _Requirements: 13.3, 13.4, 13.8_

  - [x]* 16.8 Write property test for DataSource mode-agnosticism
    - **Property 9: DataSource mode-agnosticism**
    - **Validates: Requirements 13.1, 13.7**

  - [x]* 16.9 Write property test for source_mode audit integrity
    - **Property 10: source_mode audit integrity**
    - **Validates: Requirements 13.2**

  - [x]* 16.10 Write property test for mode hand-off monotonicity
    - **Property 11: Mode hand-off monotonicity**
    - **Validates: Requirements 13.3, 13.8**

- [x] 17. Checkpoint - dual-mode data layer
  - Ensure all tests pass, ask the user if questions arise.

- [x] 18. Reconcile Phase 3 exact API response shapes
  - [x] 18.1 Implement `api/response_adapters.py`
    - Implement `ResponseAdapter` with `predict_next`, `history`, `drift_events`, and `dashboard` producing the exact Model 5 shape `{accuracy, lastSynced, timeline, decisions, driftEvents}`, computing `hit == (predicted == actual)`, ordering arrays ascending, and returning empty arrays (not null) when there is no data, reusing existing internal predict/retrain/drift logic unchanged
    - _Requirements: 14.1, 14.2_

  - [x] 18.2 Define exact-shape pydantic response models
    - Add/align pydantic response models in `api/schemas.py` to validate the exact Model 5 shape (accuracy ∈ [0,1], ISO-8601 `lastSynced`, timeline/decision/drift item fields) and bind them to the endpoint responses
    - _Requirements: 14.1_

  - [x] 18.3 Align the four `{user_id}` endpoints in `api/main.py`
    - Wire `GET /predict_next_decision/{user_id}` (`{predicted, confidence}`, `predicted` ∈ option set, confidence ∈ [0,1]), `GET /history/{user_id}` (`decisions` array), `GET /drift_events/{user_id}` (`driftEvents` array), and `POST /retrain/{user_id}` (`status` ∈ {completed, skipped} + `metrics`) through the `ResponseAdapter`; return HTTP 409 for an untrained domain without mutating state, and a `skipped`/`insufficient_data` retrain result that keeps serving the prior artifact
    - _Requirements: 14.3, 14.4, 14.5, 14.6, 14.7, 14.8_

  - [x]* 18.4 Write property test for API response-shape conformance
    - **Property 13: API response-shape conformance**
    - **Validates: Requirements 14.1**

  - [x]* 18.5 Write integration tests for the exact-shape endpoints
    - Test each endpoint and the consolidated dashboard against the pydantic schema, empty-state arrays, the 409 untrained-domain path, and the insufficient-data skip
    - _Requirements: 14.2, 14.4, 14.8_

- [ ] 19. Checkpoint - Phase 3 API shapes
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 20. Implement the React full-motion dashboard (Phase 4)
  - [ ] 20.1 Scaffold the `frontend/` project
    - Create `frontend/package.json` (React, recharts, lucide-react, Tailwind, framer-motion), `frontend/index.html`, `frontend/tailwind.config.js`, and the build/dev setup so the app compiles and runs
    - _Requirements: 17.2_

  - [ ] 20.2 Implement `frontend/src/BehavioralTwinDashboard.tsx` structure and data
    - Single-file default-export component rendering Header (twin status, `lastSynced`, accuracy vital sign), The Mirror (real vs predicted + confidence band), Decision Timeline (hit/miss), and Drift Alerts Panel from the exact Dashboard_Response; fetch from the Phase 3 API and fall back to a `MOCK_DATA` object of the same shape on error/timeout; no `localStorage`
    - _Requirements: 15.2, 15.3, 15.4_

  - [ ] 20.3 Implement the 120fps motion budget
    - Animate only `transform`/`opacity`; draw the Mirror line via precomputed-path-length `stroke-dashoffset`; heartbeat as a pure CSS keyframe (no timer/state loop); any JS motion via `requestAnimationFrame`; `will-change` on at most 2–3 continuously animating elements; memoize static sections; honor `prefers-reduced-motion` with a final-state fallback
    - _Requirements: 15.1, 15.6, 15.7_

  - [ ] 20.4 Make the dashboard responsive
    - Ensure all four sections render without horizontal scrolling or clipping across viewport widths 320–1920px
    - _Requirements: 15.5_

  - [ ]* 20.5 Write frontend motion-audit and contract tests
    - **Property 17: Frontend motion-budget conformance**
    - Also test the API-success render and the MOCK_DATA fallback render
    - **Validates: Requirements 15.1**

- [ ] 21. Checkpoint - React dashboard
  - Ensure the frontend builds and renders against the API and the mock fallback, ask the user if questions arise.

- [ ] 22. Implement the Digital Twin chat (Phase 4.5)
  - [ ] 22.1 Implement `api/chat.py` GeminiChatService
    - Implement `ChatContext` and `GeminiChatService.build_context` (read-only: current prediction, confidence ∈ [0,1], most-recent-K history, behavior summary, drift score) and `ask(user_id, question)` returning prose only, embedding the structured context in every prompt, reading the key from `GEMINI_API_KEY` (never hardcoded), and never mutating any prediction/model/stored decision
    - _Requirements: 16.1, 16.2, 16.4_

  - [ ] 22.2 Implement the response cache
    - Cache answers per `(question, context)` pair, return cached answers without a new API call, and bound the cache to the 1000 most recent distinct pairs
    - _Requirements: 16.3_

  - [ ] 22.3 Implement fallbacks and the prediction-integrity guard
    - On Gemini error, >10s timeout, or missing key, return a fallback message and leave all predictions/state unchanged; if a Gemini response asserts a decision/confidence differing from the context, return the ML model's values unchanged and present Gemini content only as prose
    - _Requirements: 16.5, 16.6_

  - [ ] 22.4 Wire a chat endpoint into the API
    - Add a chat endpoint that invokes `GeminiChatService.ask` and returns the prose answer, disabled with a clear message when the key is missing
    - _Requirements: 16.1, 16.5_

  - [ ]* 22.5 Write tests for the explain-only separation
    - **Property 12: Gemini-never-predicts separation**
    - Snapshot prediction/model/store before and after `ask` and assert equality; assert every prompt embeds structured context; assert repeated `(question, context)` pairs hit the cache; assert missing-key fallback
    - **Validates: Requirements 16.1**

- [ ] 23. Checkpoint - twin chat
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 24. Implement packaging and deployment (Phase 5)
  - [ ] 24.1 Write `Dockerfile.api` and `Dockerfile.frontend`
    - Containerize the FastAPI backend and the React frontend with reproducible builds
    - _Requirements: 17.1_

  - [ ] 24.2 Write `docker-compose.yml`
    - Compose the API and frontend so a single command starts both and makes each reachable on its configured port; fail with a non-zero exit and a clear error if a service does not become reachable within the startup window
    - _Requirements: 17.1, 17.4_

  - [ ] 24.3 Write `README.md` and finalize repo layout
    - README with an ASCII architecture diagram, setup steps for the single Docker Compose command, and model results (Accuracy, macro-F1, Brier_Score); ensure the repo layout includes `/data`, `/models`, `/api`, `/frontend`, and `/notebooks`
    - _Requirements: 17.2, 17.3_

- [ ] 25. Implement MLOps and production monitoring (Phase 6)
  - [ ] 25.1 Implement `mlops/logging_config.py` (structured logging)
    - Configure structured logging used across backend modules and replace any `print` statements in the backend with structured log calls
    - _Requirements: 18.7_

  - [ ] 25.2 Implement `mlops/prediction_log.py`
    - Implement `PredictionLogger.log` writing exactly one `PredictionLogEntry` per served prediction (prediction, confidence ∈ [0,1], latency ms, model version, ISO-8601 timestamp, user id) and `attach_actual` setting `actual` at most once; logging failures must not block the response and must emit a structured error
    - _Requirements: 18.1, 18.9_

  - [ ] 25.3 Implement `mlops/registry.py` and `mlops/tracking.py` (MLflow)
    - Implement an MLflow-backed `ModelRegistry` (register/active/promote/previous with exactly one active version per domain) and experiment tracking that records registration metrics
    - _Requirements: 18.3, 18.5_

  - [ ] 25.4 Implement `mlops/drift_monitor.py`
    - Implement `DriftMonitor.append`/`series` as an append-only, timestamp-ordered series that never mutates or drops prior points
    - _Requirements: 18.2_

  - [ ] 25.5 Implement `mlops/retrain_trigger.py`
    - Implement `RetrainTrigger.should_retrain` firing when the schedule is due or the most recent drift score exceeds the threshold, and otherwise not firing
    - _Requirements: 18.4_

  - [ ] 25.6 Implement `mlops/rollback.py`
    - Implement `RollbackController` promoting a candidate only if it does not underperform the baseline within tolerance (Accuracy/macro-F1 ≥ baseline − tol, Brier ≤ baseline + tol), otherwise keeping/restoring the baseline, maintaining one active version per domain
    - _Requirements: 18.3_

  - [ ] 25.7 Implement `mlops/metrics.py` (Prometheus)
    - Expose latency, error-rate, and throughput metrics via Prometheus exporters
    - _Requirements: 18.8_

  - [ ] 25.8 Implement `api/health.py`
    - Expose a health/readiness endpoint that reports ready only when the required model artifact for each served domain is loaded
    - _Requirements: 18.6_

  - [ ] 25.9 Wire logging, metrics, and drift-over-time into the serving path
    - Integrate `PredictionLogger`, `metrics`, and `DriftMonitor` into the prediction flow and the retrain/promote flow (via `RetrainTrigger`/`RollbackController`) so production drift is tracked over time and retrains promote/rollback safely
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.8_

  - [ ]* 25.10 Write property test for prediction-log completeness
    - **Property 14: Prediction-log completeness**
    - **Validates: Requirements 18.1**

  - [ ]* 25.11 Write property test for drift-over-time monotonic logging
    - **Property 15: Drift-over-time monotonic logging**
    - **Validates: Requirements 18.2**

  - [ ]* 25.12 Write property test for rollback safety
    - **Property 16: Rollback safety**
    - **Validates: Requirements 18.3**

- [ ] 26. Final checkpoint - full system
  - Ensure the backend builds and all tests pass, the frontend builds and renders (API + mock), Docker Compose brings up API + frontend together, and every requirement clause (1.1–18.9) is covered; ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster MVP; all other sub-tasks are core implementation.
- Property tests (Properties 1–17) validate the universal correctness properties from the design using `hypothesis` (and a frontend motion audit for Property 17) and are placed next to the code they validate to catch errors early. Unit and integration tests cover specific examples, edge cases, and end-to-end flows.
- Each task references granular requirement clauses for traceability; every requirement clause (1.1–18.9) is covered by at least one implementation task.
- Tasks 1.1, 1.2, and 2.1 are already implemented in the repository and are marked complete. The original core (data layer, feature pipeline, baseline + sequence models, evaluation, FastAPI predict/retrain, personalization, drift detection, demos) is implemented under tasks 1–14; tasks 15+ extend the system to the full master-prompt scope.
- The API is stood up first as a model-serving core (task 9), then enhanced with per-user profile features and drift status (task 10), honoring the requested "API → personalization/drift" order while ensuring each step integrates cleanly with no orphaned code.
- Tasks 16–26 follow the master-prompt phase order: Phase 1 dual-mode data layer (16), Phase 3 exact API shapes (18), Phase 4 React dashboard (20), Phase 4.5 Gemini chat (22), Phase 5 packaging (24), Phase 6 MLOps (25), with checkpoints between phases and a final full-system checkpoint (26).
- The dual-mode `DataSource` layer (task 16) sits behind the existing `DecisionStore`/`FeaturePipeline` boundary, so the feature and model code remains mode-unaware; the `source_mode` column is additive with a schema-version bump.
- The Gemini chat layer (task 22) is strictly separated from prediction: the ML model is the only source of predictions and Gemini is used only for prose explanation, guarded by tests.
- The design specifies Python for the backend; the Phase 4 frontend (task 20) is a single-file React/TypeScript component.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.3", "2.2", "2.3", "2.4", "4.1", "4.2", "4.3", "10.1", "10.4"] },
    { "id": 1, "tasks": ["2.5", "2.6", "4.4"] },
    { "id": 2, "tasks": ["4.5", "4.6", "6.1", "9.1", "10.2"] },
    { "id": 3, "tasks": ["6.2", "8.1", "10.3", "10.5"] },
    { "id": 4, "tasks": ["6.3", "6.4", "6.6", "8.2"] },
    { "id": 5, "tasks": ["6.5", "8.3", "9.2"] },
    { "id": 6, "tasks": ["9.3", "10.6"] },
    { "id": 7, "tasks": ["10.7", "12.1", "12.2", "12.3", "13.1"] },
    { "id": 8, "tasks": ["12.4", "13.2", "14.1", "14.2", "16.1"] },
    { "id": 9, "tasks": ["16.2", "16.4", "16.5"] },
    { "id": 10, "tasks": ["16.3", "16.6", "16.9"] },
    { "id": 11, "tasks": ["16.7", "16.8", "16.10"] },
    { "id": 12, "tasks": ["18.1", "18.2"] },
    { "id": 13, "tasks": ["18.3", "18.4", "18.5"] },
    { "id": 14, "tasks": ["20.1", "22.1", "25.1"] },
    { "id": 15, "tasks": ["20.2", "20.3", "22.2", "22.3", "25.2", "25.3", "25.4", "25.5"] },
    { "id": 16, "tasks": ["20.4", "20.5", "22.4", "22.5", "25.6", "25.7", "25.8"] },
    { "id": 17, "tasks": ["25.9", "25.10", "25.11", "25.12", "24.1"] },
    { "id": 18, "tasks": ["24.2", "24.3"] }
  ]
}
```
