# Requirements Document

## Introduction

The Behavioral Digital Twin is an AI system that learns an individual's decision-making
patterns from historical data (initially synthetic) and predicts the next likely decision
across three domains: route choice, productivity task choice, and consumer purchase choice.
For each prediction the system returns a predicted decision and a calibrated confidence
score, and it continuously adapts to each user as new decisions arrive.

The system is delivered as a modular, demoable repository composed of cooperating modules: a
configurable synthetic data layer (generator plus CSV/SQLite store), a deterministic feature
engineering pipeline, two competing behavioral twin models (a gradient-boosting baseline and
a sequence model), a FastAPI prediction/retraining service with concept-drift detection, a
per-user personalization layer, end-to-end demo use cases, a Streamlit dashboard, and an
evaluation/ethics report. Layer boundaries are defined by stable interfaces and a shared,
versioned data schema so the synthetic data source can later be replaced by real data without
touching the modeling, serving, or UI layers.

The system additionally provides a dual-mode data layer (a mode-agnostic `DataSource`
contract fronting a `SyntheticDataSource` and a `RealDataSource`, governed by a per-user
`ModeManager` with a synthetic-to-real migration path) that stamps a `source_mode` field on
every stored record; a Phase 3 serving layer whose consolidated response shape conforms
exactly to the React contract via a `ResponseAdapter`; a single-file React full-motion
dashboard as the primary UI deliverable; a Gemini-backed Digital Twin chat
(`GeminiChatService`) used strictly for explanation and never for prediction; Docker-based
packaging to run the API and frontend together; and an MLOps platform for model versioning,
prediction logging, production drift monitoring, automatic retraining, rollback, and
performance monitoring.

These requirements are derived from the approved design document and are organized so that
every correctness property in the design maps to one or more acceptance criteria below.

## Glossary

- **System**: The complete Behavioral Digital Twin application across all modules.
- **Data_Generator**: The `SyntheticDataGenerator` component that produces configurable
  synthetic decision data (`data/synthetic_data_generator.py`).
- **Decision_Store**: The `DecisionStore` component that persists and loads decision records
  via a CSV or SQLite backend (`data/decision_store.py`).
- **Decision_Record**: A single observed decision with context, choice, and outcome fields.
- **Feature_Pipeline**: The deterministic component that transforms decision records into
  model-ready feature vectors/matrices (`features/feature_pipeline.py`).
- **Baseline_Model**: The gradient-boosting / logistic-regression `DecisionModel`.
- **Sequence_Model**: The LSTM / Transformer `DecisionModel` consuming last-K history.
- **Evaluator**: The component that scores models using Accuracy, macro-F1, and Brier.
- **Prediction_Engine**: The FastAPI service exposing prediction, profile, and retrain
  endpoints (`api/main.py`).
- **Profile_Store**: The `UserProfileStore` holding per-user embeddings and statistics.
- **Profile_Updater**: The `ProfileUpdater` applying online (EMA) profile updates.
- **Drift_Detector**: The component that flags concept drift from rolling accuracy.
- **Dashboard**: The Streamlit visualization application (`dashboard/app.py`).
- **Domain**: One of `route`, `task`, or `purchase`.
- **Option_Set**: The set of valid decision options for a given domain.
- **Context**: The prediction-time input describing situation features for a decision.
- **K**: The fixed history window length (number of most recent decisions used as sequence).
- **PAD**: The reserved padding identifier used to left-pad short history windows.
- **UNK**: The explicit bucket for categorical values unseen during pipeline fit.
- **EMA**: Exponential moving average used for online embedding updates.
- **Brier_Score**: The mean squared error between predicted class probabilities and outcomes.
- **Drift_Status**: A result object containing `drift` (boolean), `score`, and `window_acc`.
- **Source_Mode**: The audit field stamped on every Decision_Record, with value `synthetic`
  or `real`, recording which mode produced the row.
- **Data_Source**: The mode-agnostic contract (`DataSource`) for fetching decision records,
  implemented by Synthetic_Data_Source and Real_Data_Source.
- **Synthetic_Data_Source**: The Mode 1 (Demo) `DataSource` implementation that wraps the
  Data_Generator and tags records `synthetic`.
- **Real_Data_Source**: The Mode 2 (Production) `DataSource` implementation that merges
  Source_Connector output and tags records `real`.
- **Source_Connector**: A single real-data connector behind the same contract; either a
  concrete connector (user interactions, route selections, CSV import) or a same-contract
  stub (study sessions, productivity logs, calendar, GitHub, browser opt-in, weather, time,
  device).
- **Mode_Manager**: The component holding each user's mode flag/config and governing the
  synthetic-to-real hand-off (`ModeManager`).
- **Migration_Threshold**: The real-record count at which a user is handed off to real mode.
- **Response_Adapter**: The component (`ResponseAdapter`) that maps internal results into the
  exact frontend-facing response shapes.
- **Dashboard_Response**: The consolidated, exact-shape Phase 3 response consumed by the
  React dashboard (Model 5).
- **React_Dashboard**: The single-file React full-motion dashboard component
  (`BehavioralTwinDashboard`) that is the primary UI deliverable.
- **Mock_Data**: The fallback dataset, identical in shape to the Dashboard_Response, rendered
  by the React_Dashboard when the API is unreachable.
- **Gemini_Chat_Service**: The conversational component (`GeminiChatService`) that uses the
  Gemini API strictly for explanation, summarization, and comparison.
- **ML_Model**: The trained behavioral twin model; the single source of predictions.
- **Prediction_Logger**: The MLOps component that records one log entry per served prediction.
- **Model_Registry**: The MLflow-backed component that versions and tracks trained models.
- **Drift_Monitor**: The MLOps component that tracks production drift over time as an
  append-only series.
- **Retrain_Trigger**: The component that initiates retraining on schedule or on a drift
  threshold.
- **Rollback_Controller**: The component that promotes a candidate model only if it does not
  underperform the active baseline, and otherwise keeps or restores the baseline.

## Requirements

### Requirement 1: Configurable Synthetic Data Generation

**User Story:** As a developer, I want a configurable synthetic data generator, so that I
can produce realistic, reproducible behavioral datasets for development and demos.

#### Acceptance Criteria

1. WHEN the Data_Generator is invoked with a valid `GeneratorConfig`, THE Data_Generator SHALL produce decision records spanning the configured number of days with the configured per-domain habit patterns, weekday/weekend differences, gradual drift, and bounded random noise.
2. WHEN the Data_Generator produces a record sequence, THE Data_Generator SHALL return the records sorted ascending by timestamp.
3. WHEN the Data_Generator is invoked two or more times with the same `GeneratorConfig` including the same seed, THE Data_Generator SHALL produce an identical record sequence on every run.
4. IF the `GeneratorConfig` contains an invalid value, where `n_days` is not greater than zero or `weekend_shift`, `drift_rate`, or `noise` is outside the range zero to one inclusive or `domains` is empty or not a subset of the supported domains, THEN THE Data_Generator SHALL reject the configuration and return a descriptive validation error.
5. THE Data_Generator SHALL provide a method to return the generated records as a pandas DataFrame conforming to the shared column schema.

### Requirement 2: Decision Record Schema Validity

**User Story:** As a developer, I want a shared, validated decision schema, so that every
record is well-formed and consistent across the data, feature, and model layers.

#### Acceptance Criteria

1. WHERE a Decision_Record is generated or stored, THE System SHALL ensure the `domain` field is one of `route`, `task`, or `purchase`.
2. WHERE a Decision_Record is generated or stored, THE System SHALL ensure the `decision_made` value belongs to the Option_Set of that record's domain.
3. WHERE a Decision_Record is generated or stored, THE System SHALL ensure the `mood_energy` value is within the range zero to one inclusive.
4. WHERE a Decision_Record is generated or stored, THE System SHALL ensure the `time_of_day` value is derived consistently from the record timestamp hour.
5. WHERE a Decision_Record is generated or stored, THE System SHALL ensure the `day_type` value is derived consistently from the record timestamp weekday.

### Requirement 3: Decision Persistence

**User Story:** As a developer, I want a stable persistence boundary, so that synthetic data
can be replaced by real data without changing downstream layers.

#### Acceptance Criteria

1. WHERE the Decision_Store is configured with the `csv` or `sqlite` backend, THE Decision_Store SHALL persist appended decision records to the configured path through one interface.
2. WHEN records are appended to the Decision_Store, THE Decision_Store SHALL retain all appended records for later retrieval.
3. WHEN records are loaded from the Decision_Store, THE Decision_Store SHALL return the matching records sorted ascending by timestamp.
4. WHEN the Decision_Store is queried with a `user_id` or `since` filter, THE Decision_Store SHALL return only records matching the supplied filter.
5. WHEN the Decision_Store `count` operation is invoked, THE Decision_Store SHALL return the number of stored records matching the supplied filter.

### Requirement 4: Deterministic Feature Engineering

**User Story:** As a data scientist, I want a deterministic feature pipeline, so that models
are trained on leakage-free, consistently encoded features.

#### Acceptance Criteria

1. WHEN the Feature_Pipeline transforms a record set after being fit, THE Feature_Pipeline SHALL build temporal features including hour encoding, day-of-week encoding, and rolling frequency for each record.
2. WHEN the Feature_Pipeline transforms a record set, THE Feature_Pipeline SHALL build a last-K decision sequence for each record, left-padded with the PAD identifier to length K.
3. WHEN the Feature_Pipeline builds the history component for the record at index i, THE Feature_Pipeline SHALL include only decisions strictly earlier than record i.
4. WHEN the Feature_Pipeline builds the rolling frequency component for the record at index i, THE Feature_Pipeline SHALL include only decisions strictly earlier than record i.
5. WHEN the Feature_Pipeline encodes categorical context, THE Feature_Pipeline SHALL apply the same encoding learned during fit and produce fixed, schema-versioned dimensions.
6. IF a categorical value unseen during fit is encountered during transform, THEN THE Feature_Pipeline SHALL map that value to the explicit UNK bucket without raising an error.

### Requirement 5: Behavioral Twin Models

**User Story:** As a data scientist, I want two competing model families with probabilistic
output, so that I can predict the next decision and report calibrated confidence.

#### Acceptance Criteria

1. WHEN the Baseline_Model is fit on flat feature vectors and labels, THE Baseline_Model SHALL learn to predict the next decision for the domain.
2. WHEN the Sequence_Model is fit on the last-K sequence with context and user embedding, THE Sequence_Model SHALL learn to predict the next decision for the domain.
3. WHEN either model produces a prediction, THE System SHALL ensure the returned `class_probs` are non-negative and sum to approximately one.
4. WHEN either model produces a prediction, THE System SHALL ensure `predicted_decision` belongs to the domain Option_Set and `confidence` equals the maximum value in `class_probs`.
5. WHEN a model is saved and then loaded from an artifact path, THE System SHALL produce a loaded model that yields predictions equal to the saved model for identical inputs.

### Requirement 6: Time-Based Split and Model Comparison

**User Story:** As a data scientist, I want a strictly temporal evaluation, so that model
comparison reflects realistic forecasting without future leakage.

#### Acceptance Criteria

1. WHEN `time_based_split` is applied to a sorted record set with a valid `val_fraction`, THE Evaluator SHALL produce a partition of the input with no lost and no duplicated records.
2. WHEN `time_based_split` partitions a record set, THE Evaluator SHALL ensure every train timestamp is less than or equal to every validation timestamp.
3. WHEN model comparison runs on the validation partition, THE Evaluator SHALL compute Accuracy, macro-F1, and Brier_Score for both the Baseline_Model and the Sequence_Model.
4. WHEN model comparison completes, THE Evaluator SHALL produce a report containing the computed metrics for both models.

### Requirement 7: Prediction Engine API

**User Story:** As a client developer, I want HTTP endpoints for prediction, profiles, and
retraining, so that I can integrate the behavioral twin into applications and demos.

#### Acceptance Criteria

1. WHEN a client sends a valid request to `POST /predict_next_decision` with `user_id`, `context`, and `recent_decisions`, THE Prediction_Engine SHALL return a response containing `predicted_decision`, `confidence`, `class_probs`, and `drift_status`.
2. WHEN the Prediction_Engine returns a prediction response, THE Prediction_Engine SHALL ensure `class_probs` are non-negative and sum to approximately one.
3. WHEN the Prediction_Engine returns a prediction response, THE Prediction_Engine SHALL ensure `predicted_decision` belongs to the requested domain Option_Set and `confidence` equals the maximum value in `class_probs`.
4. WHEN a client sends a request to `GET /user_profile/{id}`, THE Prediction_Engine SHALL return the user's decision counts, embedding summary, and last-updated timestamp.
5. WHEN a client sends a request to `POST /retrain`, THE Prediction_Engine SHALL retrain on the requested records and return a status and the evaluation metrics.
6. IF `POST /predict_next_decision` is called for a domain whose model artifact has not been trained, THEN THE Prediction_Engine SHALL respond with HTTP status 409 and a message directing the client to run `/retrain`.
7. IF `POST /retrain` is requested with too few records for a valid temporal split, THEN THE Prediction_Engine SHALL respond with HTTP status 200, a status of `skipped`, and the reason `insufficient_data`, and SHALL continue serving the previously trained artifact.

### Requirement 8: Demo Use Cases

**User Story:** As an evaluator, I want runnable demos for each domain, so that I can see the
behavioral twin produce predictions end to end.

#### Acceptance Criteria

1. WHEN the route demo is executed, THE System SHALL produce a next-route prediction with confidence for a sample user using the Prediction_Engine.
2. WHEN the productivity demo is executed, THE System SHALL produce a next-task prediction with confidence for a sample user using the Prediction_Engine.
3. WHEN the purchase demo is executed, THE System SHALL produce a next-purchase prediction with confidence for a sample user using the Prediction_Engine.
4. WHERE a demo is executed, THE System SHALL run the full path of generate, store, feature build, train, and predict without manual intervention.

### Requirement 9: Visualization Dashboard

**User Story:** As an evaluator, I want a dashboard, so that I can inspect a user's decisions,
prediction quality, confidence trends, and drift alerts.

#### Acceptance Criteria

1. WHEN the Dashboard loads a user's decisions, THE Dashboard SHALL display a decision timeline ordered ascending by timestamp.
2. WHEN the Dashboard displays prediction quality, THE Dashboard SHALL show predicted decisions alongside actual decisions for comparison.
3. WHEN the Dashboard displays prediction history, THE Dashboard SHALL show confidence over time.
4. WHEN the Drift_Detector reports a drift condition, THE Dashboard SHALL display a drift alert.

### Requirement 10: Concept-Drift Detection

**User Story:** As a system operator, I want concept-drift detection, so that I am alerted
when a user's behavior diverges from the twin's predictions.

#### Acceptance Criteria

1. WHEN a labeled prediction is recorded with a predicted decision, actual decision, and confidence, THE Drift_Detector SHALL update its rolling record of recent labeled predictions.
2. WHEN the Drift_Detector status is requested with no labeled predictions observed, THE Drift_Detector SHALL report `drift` as false and a window accuracy of none.
3. WHEN the Drift_Detector status is requested, THE Drift_Detector SHALL report `drift` as true if and only if at least `window` labeled predictions exist and the rolling accuracy is less than the configured threshold.
4. WHEN the Drift_Detector status is requested with at least one labeled prediction, THE Drift_Detector SHALL compute `window_acc` as the proportion of correct predictions over the most recent `window` labeled predictions.

### Requirement 11: Per-User Personalization

**User Story:** As a user, I want my twin to adapt to my recent decisions, so that
predictions reflect my evolving behavior without full retraining.

#### Acceptance Criteria

1. WHEN the Prediction_Engine builds features for a prediction, THE Profile_Store SHALL supply the current per-user embedding and statistics for the requested user.
2. WHEN the Profile_Updater processes new records for a user, THE Profile_Updater SHALL increment each decision count by the occurrences in the new records so that no count decreases below its prior value.
3. WHEN the Profile_Updater processes new records for a user, THE Profile_Updater SHALL move the user embedding toward the aggregate behavior of the new records using an EMA update.
4. WHEN the Profile_Updater processes new records for a user, THE Profile_Updater SHALL set `last_updated` to the maximum timestamp among the new records so that `last_updated` is non-decreasing.

### Requirement 12: Evaluation and Ethics Documentation

**User Story:** As an evaluator, I want documented evaluation and ethics analysis, so that I
understand the system's performance, privacy posture, and misuse mitigations.

#### Acceptance Criteria

1. THE System SHALL provide a README documenting setup, how to generate data, train, serve the API, run demos, and launch the dashboard.
2. THE System SHALL provide a REPORT.md documenting model comparison results across Accuracy, macro-F1, and Brier_Score.
3. THE System SHALL document in REPORT.md the privacy posture including synthetic-by-default data, data minimization, user consent, per-user isolation, and a user-controlled delete/export path.
4. THE System SHALL document in REPORT.md the misuse mitigations covering manipulation and surveillance risks, prediction transparency, user control, and opt-out.
5. WHERE the demo API is deployed without authentication, THE System SHALL explicitly flag that authentication and per-user authorization are required before any non-demo exposure.

### Requirement 13: Dual-Mode Data Layer

**User Story:** As a system architect, I want a mode-agnostic data layer with synthetic and
real implementations behind one contract, so that the system can migrate a user from
synthetic to real data without changing the feature, model, or serving layers.

#### Acceptance Criteria

1. WHERE the active Data_Source resolved for a user is the Synthetic_Data_Source or the Real_Data_Source, THE Data_Source SHALL expose the same mode-agnostic contract whose `fetch` operation returns schema-valid Decision_Records sorted ascending by timestamp and whose `count` operation returns the non-negative integer number of records matching the supplied filter.
2. WHERE a Decision_Record is produced through the Data_Source layer and persisted, THE System SHALL stamp the record's Source_Mode field with `synthetic` when produced by the Synthetic_Data_Source and `real` when produced by the Real_Data_Source, so that the stored Source_Mode value equals the mode of the source that produced the record.
3. WHEN the Mode_Manager evaluates a user whose real-record count is greater than or equal to that user's Migration_Threshold, where the Migration_Threshold is a positive integer count, THE Mode_Manager SHALL set the user's active mode to real.
4. THE Mode_Manager SHALL default a new user's active mode to synthetic.
5. WHERE the Real_Data_Source aggregates Source_Connectors, THE Real_Data_Source SHALL provide concrete connectors for user interactions, route selections, and CSV import, and SHALL provide same-contract stub connectors for study sessions, productivity logs, calendar, GitHub activity, opt-in browser activity, weather, time, and device that each return an empty but schema-valid record set and report themselves as disabled until implemented.
6. IF a Source_Connector errors or reports itself as disabled, THEN THE Real_Data_Source SHALL skip that connector, log a structured warning indicating the skipped connector, and continue returning schema-valid records aggregated from the remaining connectors.
7. WHEN the Feature_Pipeline or a model consumes records obtained through the Data_Source contract, THE System SHALL produce outputs of the same shape and dimensions regardless of whether the resolved source was the Synthetic_Data_Source or the Real_Data_Source, and SHALL NOT branch on the active mode.
8. WHEN the Mode_Manager evaluates a user across successive evaluations, THE Mode_Manager SHALL keep the active mode non-decreasing such that a user in real mode never regresses to synthetic mode, and SHALL keep the blend weight within the range zero to one inclusive and non-decreasing as the user's real-record count grows.

### Requirement 14: Phase 3 Exact API Response Shapes

**User Story:** As a frontend developer, I want endpoints and a consolidated response whose
shape is exact and stable, so that the React dashboard can consume the data without
defensive parsing.

#### Acceptance Criteria

1. WHEN the Response_Adapter produces the consolidated Dashboard_Response for a user, THE Response_Adapter SHALL return an object that matches the exact Model 5 shape `{accuracy, lastSynced, timeline[{date, actual, predicted, confidence}], decisions[{id, timestamp, domain, predicted, actual, hit, confidence}], driftEvents[{date, domain, note}]}`, with `accuracy` within the range zero to one inclusive, each `timeline` entry's `confidence` within the range zero to one inclusive, `lastSynced` formatted as an ISO-8601 timestamp, `timeline` ordered ascending by `date`, `decisions` ordered ascending by `timestamp`, `driftEvents` ordered ascending by `date`, and for every entry in `decisions` the `hit` value equal to the result of comparing `predicted` to `actual`.
2. WHEN the Response_Adapter produces the consolidated Dashboard_Response for a user who has no decisions, no timeline points, and no drift events, THE Response_Adapter SHALL return the `timeline`, `decisions`, and `driftEvents` fields as empty arrays rather than null or absent.
3. WHEN a client sends a request to `GET /predict_next_decision/{user_id}`, THE Prediction_Engine SHALL return an object containing `predicted` and `confidence` derived from the internal prediction result, where `predicted` belongs to the requested domain Option_Set and `confidence` is within the range zero to one inclusive.
4. IF a client sends a request to `GET /predict_next_decision/{user_id}` for a domain whose model artifact has not been trained, THEN THE Prediction_Engine SHALL respond with HTTP status 409 and an error message directing the client to run retraining, and SHALL NOT modify any stored state.
5. WHEN a client sends a request to `GET /history/{user_id}`, THE Prediction_Engine SHALL return the `decisions` array conforming to the Model 5 decision shape, ordered ascending by `timestamp`, and SHALL return an empty array when the user has no decisions.
6. WHEN a client sends a request to `GET /drift_events/{user_id}`, THE Prediction_Engine SHALL return the `driftEvents` array conforming to the Model 5 drift-event shape, ordered ascending by `date`, and SHALL return an empty array when the user has no drift events.
7. WHEN a client sends a request to `POST /retrain/{user_id}`, THE Prediction_Engine SHALL retrain using the existing retrain logic and return an object containing a `status` whose value is one of `completed` or `skipped` and a `metrics` object containing Accuracy, macro-F1, and Brier_Score.
8. IF a client sends a request to `POST /retrain/{user_id}` with too few records for a valid temporal split, THEN THE Prediction_Engine SHALL respond with a `status` of `skipped` and the reason `insufficient_data`, and SHALL continue serving the previously trained artifact.

### Requirement 15: React Full-Motion Dashboard

**User Story:** As an evaluator, I want a smooth, single-file React dashboard, so that I can
view the twin's status, predictions, decision history, and drift alerts with high-performance
motion.

#### Acceptance Criteria

1. WHEN the React_Dashboard renders, THE React_Dashboard SHALL animate only the `transform` and `opacity` CSS properties, draw the SVG series line by animating `stroke-dashoffset` from a precomputed path length, render the heartbeat indicator as a pure CSS keyframe animation without any timer or state-update loop, drive any JavaScript-based motion through `requestAnimationFrame`, apply `will-change` to at most three continuously animating elements at any one time, and memoize static sections so that they do not re-render when only animated values change.
2. WHEN the React_Dashboard receives the Dashboard_Response, THE React_Dashboard SHALL render the Header showing twin status, `lastSynced`, and the accuracy vital sign; the Mirror showing real versus predicted values with a confidence band; the Decision Timeline showing hit and miss outcomes; and the Drift Alerts Panel, all populated from the exact Phase 3 Dashboard_Response without altering any field value.
3. IF the API does not return the Dashboard_Response within 5 seconds or returns an error response, THEN THE React_Dashboard SHALL render all four sections from the Mock_Data fallback, which has the same shape as the Dashboard_Response.
4. THE React_Dashboard SHALL be a single-file default-export component that does not use `localStorage`.
5. WHILE rendered on a viewport width from 320 pixels to 1920 pixels inclusive, THE React_Dashboard SHALL display the Header, the Mirror, the Decision Timeline, and the Drift Alerts Panel without horizontal scrolling and without content clipped outside the viewport.
6. IF the user agent reports `prefers-reduced-motion: reduce`, THEN THE React_Dashboard SHALL disable the `transform`, `opacity`, `stroke-dashoffset`, and heartbeat animations and SHALL render each section directly in its final visual state.
7. WHILE continuous animations are running on a viewport within the 320-pixel to 1920-pixel range, THE React_Dashboard SHALL sustain a target rendering rate of 60 frames per second.

### Requirement 16: Digital Twin Chat (Explain-Only)

**User Story:** As a user, I want to ask my twin to explain its behavior, so that I can
understand predictions in natural language without those explanations ever changing what is
predicted.

#### Acceptance Criteria

1. WHEN the Gemini_Chat_Service answers a question, THE Gemini_Chat_Service SHALL leave the active predicted decision, the confidence value, the model artifacts, and the stored Decision_Records unchanged, SHALL use the Gemini response solely as explanatory prose that does not modify any predicted decision or confidence, and SHALL include the structured prediction context in every Gemini prompt, so that the ML_Model remains the only source of predictions.
2. WHEN the Gemini_Chat_Service builds a prompt, THE Gemini_Chat_Service SHALL include, as read-only structured context, the current predicted decision, the confidence as a value in the range zero to one inclusive, the most recent K decisions from history, the per-user behavior summary, and the current drift score.
3. WHEN the Gemini_Chat_Service receives a question whose `(question, context)` pair matches a previously answered pair, THE Gemini_Chat_Service SHALL return the stored response for that pair without issuing a new Gemini API call, retaining at most the 1000 most recent distinct `(question, context)` pairs.
4. THE Gemini_Chat_Service SHALL read the Gemini API key from an environment variable or secret store and SHALL NOT hardcode the key in source.
5. IF the Gemini API returns an error, does not respond within 10 seconds, or the API key is missing, THEN THE Gemini_Chat_Service SHALL return a fallback message indicating that the explanation service is temporarily unavailable and SHALL leave the active predicted decision, the confidence, the model artifacts, and the stored Decision_Records unchanged.
6. IF a Gemini response asserts a predicted decision or confidence that differs from the read-only structured context, THEN THE Gemini_Chat_Service SHALL return the ML_Model's predicted decision and confidence unchanged and SHALL present the Gemini content only as prose.

### Requirement 17: Packaging and Deployment

**User Story:** As an evaluator, I want to run the system locally with one command and clear
documentation, so that I can stand up the API and frontend together and understand the
project.

#### Acceptance Criteria

1. WHEN a single Docker Compose command is invoked, THE System SHALL start both the Prediction_Engine API and the React_Dashboard frontend together and make each reachable on its configured local port within 120 seconds.
2. THE System SHALL organize the repository with `/data`, `/models`, `/api`, `/frontend`, and `/notebooks` directories.
3. THE System SHALL provide a README containing an ASCII architecture diagram, setup steps sufficient to start the API and frontend via the single Docker Compose command, and model results reporting Accuracy, macro-F1, and Brier_Score.
4. IF a service fails to become reachable within 120 seconds of the Docker Compose command, THEN THE System SHALL terminate with a non-zero exit status and emit an error indicating which service failed to start.

### Requirement 18: MLOps and Production Monitoring

**User Story:** As a system operator, I want versioning, complete logging, monitoring,
automatic retraining, and safe rollback, so that the twin operates reliably in production and
never regresses to a worse model.

#### Acceptance Criteria

1. WHEN the Prediction_Engine serves a prediction, THE Prediction_Logger SHALL write exactly one log entry containing the predicted decision, the confidence within the range zero to one inclusive, the serving latency in milliseconds, the active model version, an ISO-8601 timestamp, and the user id, and SHALL set that entry's `actual` value at most once, when the true decision becomes known.
2. WHEN the Drift_Monitor records a production drift status update, THE Drift_Monitor SHALL append exactly one point to the user's drift series ordered by timestamp, SHALL NOT mutate previously recorded points, and SHALL retain all prior points.
3. WHEN a retrain produces a candidate model, THE Rollback_Controller SHALL promote the candidate to active only if, on the validation partition, the candidate's Accuracy and macro-F1 are each greater than or equal to the active baseline's corresponding value minus a configured tolerance and the candidate's Brier_Score is less than or equal to the active baseline's Brier_Score plus the configured tolerance, SHALL otherwise keep or restore the baseline as active, and SHALL maintain exactly one active model version per Domain at all times.
4. WHEN the Retrain_Trigger is evaluated, THE Retrain_Trigger SHALL initiate retraining if the configured schedule is due or the most recent tracked drift score exceeds the configured drift threshold, and SHALL otherwise not initiate retraining.
5. THE Model_Registry SHALL version every trained model and record its registration metrics through MLflow experiment tracking.
6. THE Prediction_Engine SHALL expose a health-check endpoint that reports the service as ready only when the required model artifact for each served Domain is loaded, and reports the service as not ready otherwise.
7. THE System SHALL emit structured logs across the backend modules and SHALL NOT use `print` statements for backend logging.
8. THE System SHALL expose latency, error-rate, and throughput metrics through Prometheus exporters.
9. IF the Prediction_Logger fails to write a prediction log entry, THEN THE Prediction_Engine SHALL still return the prediction response to the client and SHALL record a structured error indicating the logging failure.
