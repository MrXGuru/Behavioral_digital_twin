"""TwinService: orchestrates data, models, drift, and the dashboard response shape.

This is the heart of the prediction engine. For a given user it ensures decision data
exists (seeding synthetic data on first use), trains the winning model per domain on a
strictly temporal split, runs walk-forward prediction over the held-out tail to produce
predicted-vs-actual history, and assembles the exact JSON shape the React frontend
consumes:

    {accuracy, lastSynced, timeline, decisions, driftEvents}

Models and per-user state are cached in-process so repeated reads are fast; ``retrain``
rebuilds them from the latest stored data.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from api.drift import DriftDetector
from data.decision_store import DecisionStore
from data.schema import DecisionRecord, Domain, options
from features.feature_pipeline import FeaturePipeline
from models.base import build_class_probs
from models.baseline import BaselineModel
from models.evaluate import evaluate_models, time_based_split
from models.sequence import SequenceModel
from personalization.profile_store import UserProfileStore
from personalization.updater import ProfileUpdater

if TYPE_CHECKING:
    from mlops.prediction_log import PredictionLogger
    from mlops.drift_monitor import DriftMonitor
    from mlops.retrain_trigger import RetrainTrigger

_svc_logger = logging.getLogger(__name__)


class ModelNotTrainedError(Exception):
    """Raised when a prediction is requested for a domain with no trained artifact.

    The Requirement 7 ``POST /predict_next_decision`` path uses this to signal an
    HTTP 409 directing the client to run ``/retrain`` (Requirement 7.6).
    """

    def __init__(self, domain: str) -> None:
        super().__init__(domain)
        self.domain = domain

#: Pretty labels for decision options shown in the UI.
_PRETTY = {
    "pomodoro": "Pomodoro", "flow_state": "Flow State",
    "light_work": "Light Work", "admin": "Admin",
    "deep_work": "Deep Work", "email": "Email", "meeting": "Meeting", "break": "Break",
    "coffee": "Coffee", "snack": "Snack", "lunch": "Lunch", "none": "No Purchase",
}

#: Rolling window + threshold for surfacing drift events on the timeline.
_DRIFT_WINDOW = 10
_DRIFT_THRESHOLD = 0.4

DEFAULT_DOMAINS = ["focus", "task", "purchase"]

#: Minimum number of logged decisions before the twin is considered "learning"
#: enough to produce meaningful predictions.  Below this the API surfaces
#: a data_maturity == "learning" state so the UI can display an honest
#: "still learning" message instead of a polished-looking but unreliable result.
DATA_MATURITY_THRESHOLD: int = 5


def pretty(label: str) -> str:
    return _PRETTY.get(label, label)


def _iso_z(ts: datetime) -> str:
    """Serialize a datetime as UTC ISO-8601 with a trailing Z."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _DomainModel:
    """Holds the fitted pipeline + winning model + comparison metrics for a domain."""

    def __init__(self, domain: str):
        self.domain = domain
        self.pipe: FeaturePipeline | None = None
        self.model = None
        self.model_name = ""
        self.metrics: dict = {}


class TwinService:
    """Per-process behavioral-twin orchestration."""

    def __init__(self, store: DecisionStore | None = None,
                 domains: list[str] | None = None,
                 prediction_logger: "PredictionLogger | None" = None,
                 drift_monitor: "DriftMonitor | None" = None,
                 retrain_trigger: "RetrainTrigger | None" = None) -> None:
        self.store = store or DecisionStore("sqlite", "data/generated/decisions.db")
        self.domains = domains or DEFAULT_DOMAINS
        self.profiles = UserProfileStore()
        self.updater = ProfileUpdater()
        self._models: dict[str, dict[str, _DomainModel]] = {}  # user -> domain -> model
        self._reports: dict[str, list[dict]] = {}
        # Optional MLOps components — may be None; failures MUST NOT block predictions
        self._prediction_logger = prediction_logger
        self._drift_monitor = drift_monitor
        self._retrain_trigger = retrain_trigger

    # ------------------------------------------------------------------
    def ensure_user(self, user_id: str, min_records: int = 60) -> None:
        """No-op: synthetic seeding has been removed.

        Real decisions are now logged via ``POST /decisions/{user_id}``.
        This method is retained as a no-op so existing call-sites do not break.
        """
        return

    def data_maturity(self, user_id: str) -> dict:
        """Return a maturity assessment for ``user_id``.

        Returns a dict with:
          - ``count``: total number of logged decisions
          - ``threshold``: the minimum required for reliable predictions
          - ``status``: ``"learning"`` (count < threshold) or ``"ready"``
          - ``message``: a human-readable explanation

        The spec requires that below ``DATA_MATURITY_THRESHOLD`` the system shows
        "still learning" rather than a polished-looking but unreliable prediction.
        """
        count = self.store.count(user_id=user_id)
        if count < DATA_MATURITY_THRESHOLD:
            return {
                "count": count,
                "threshold": DATA_MATURITY_THRESHOLD,
                "status": "learning",
                "message": (
                    f"Still learning — only {count} decision"
                    f"{'s' if count != 1 else ''} logged so far. "
                    f"Log at least {DATA_MATURITY_THRESHOLD - count} more to unlock predictions."
                ),
            }
        return {
            "count": count,
            "threshold": DATA_MATURITY_THRESHOLD,
            "status": "ready",
            "message": f"Twin is active with {count} logged decisions.",
        }

    # ------------------------------------------------------------------
    def train(self, user_id: str) -> list[dict]:
        """Train the winning model per domain; return comparison reports."""
        records = self.store.load(user_id=user_id)
        reports: list[dict] = []
        self._models[user_id] = {}

        for domain in self.domains:
            dom_records = [r for r in records if r.domain == domain]
            if len(dom_records) < 5:
                continue
            report = evaluate_models(records, domain)
            reports.append(report.as_dict())

            train, _ = time_based_split(dom_records, 0.2)
            pipe = FeaturePipeline(domain).fit(train)
            fm = pipe.transform(train)
            opts = list(options(domain))
            if report.winner == "sequence":
                model = SequenceModel(opts).fit(fm.X, fm.seq, fm.y)
            else:
                model = BaselineModel(opts).fit(fm.X, fm.seq, fm.y)

            dm = _DomainModel(domain)
            dm.pipe, dm.model, dm.model_name = pipe, model, report.winner
            dm.metrics = report.as_dict()
            self._models[user_id][domain] = dm

        # refresh personalization profile from the full history (Req 11.1): update()
        # is side-effect free and returns a NEW profile, so persist the returned value.
        profile = self.profiles.get(user_id)
        profile = self.updater.update(profile, records)
        self.profiles.upsert(profile)
        self._reports[user_id] = reports
        return reports

    def _ensure_trained(self, user_id: str) -> None:
        if user_id not in self._models or not self._models[user_id]:
            self.train(user_id)

    # ------------------------------------------------------------------
    def _walk_forward(self, user_id: str):
        """Return per-domain (records, predicted_label, actual, prob_actual, conf)."""
        records = self.store.load(user_id=user_id)
        rows = []  # (timestamp, domain, predicted, actual, hit, confidence, prob_actual)
        for domain, dm in self._models[user_id].items():
            dom_records = [r for r in records if r.domain == domain]
            _, val = time_based_split(dom_records, 0.2)
            fm = dm.pipe.transform(val)
            for i, r in enumerate(val):
                probs = dm.model.predict_proba(fm.X[i:i + 1], fm.seq[i:i + 1])
                predicted = max(probs, key=probs.get)
                confidence = probs[predicted]
                prob_actual = probs.get(r.decision_made, 0.0)
                rows.append({
                    "timestamp": r.timestamp, "domain": domain,
                    "predicted": predicted, "actual": r.decision_made,
                    "hit": predicted == r.decision_made,
                    "confidence": confidence, "prob_actual": prob_actual,
                    "day_type": r.day_type,
                })
        rows.sort(key=lambda x: x["timestamp"])
        return rows

    # ------------------------------------------------------------------
    def dashboard(self, user_id: str) -> dict:
        """Assemble the full {accuracy, lastSynced, timeline, decisions, driftEvents}.

        Returns a clean empty-state response when the user has no data or
        insufficient data to train models, instead of raising an error.
        Includes a ``data_maturity`` field so the frontend can show an honest
        "still learning" state when the user has fewer than DATA_MATURITY_THRESHOLD
        logged decisions.
        """
        now_str = _iso_z(datetime.now(timezone.utc))
        maturity = self.data_maturity(user_id)
        empty = {"accuracy": 0.0, "lastSynced": now_str,
                 "timeline": [], "decisions": [], "driftEvents": [],
                 "data_maturity": maturity}

        # If user has no records at all, return empty state immediately.
        record_count = self.store.count(user_id=user_id)
        if record_count == 0:
            return empty

        # Even if not fully trained, we should return their logged decisions and heatmap
        records = self.store.load(user_id=user_id)
        
        # Format raw records into the expected decision shape for the UI table
        raw_decisions = [{
            "id": idx + 1,
            "timestamp": _iso_z(r.timestamp),
            "domain": r.domain,
            "predicted": "Learning...",
            "actual": r.decision_made.replace("_", " ").title(),
            "hit": False,
            "confidence": 0.0,
        } for idx, r in enumerate(records)]
        
        positive_habits = {"pomodoro", "flow_state", "deep_work"}
        heatmap_dict = {}
        for r in records:
            d_str = r.timestamp.strftime("%Y-%m-%d")
            if d_str not in heatmap_dict:
                heatmap_dict[d_str] = 0
            if r.decision_made in positive_habits:
                heatmap_dict[d_str] += 1
        heatmap = [{"date": k, "count": v} for k, v in sorted(heatmap_dict.items())]

        # Attempt to train if not already trained
        try:
            self._ensure_trained(user_id)
        except Exception:
            empty["decisions"] = raw_decisions
            empty["heatmap"] = heatmap
            return empty

        if user_id not in self._models or not self._models[user_id]:
            empty["decisions"] = raw_decisions
            empty["heatmap"] = heatmap
            return empty

        rows = self._walk_forward(user_id)
        if not rows:
            empty["decisions"] = raw_decisions
            empty["heatmap"] = heatmap
            return empty

        accuracy = round(sum(1 for r in rows if r["hit"]) / len(rows), 4)

        decisions = [{
            "id": idx + 1,
            "timestamp": _iso_z(r["timestamp"]),
            "domain": r["domain"],
            "predicted": pretty(r["predicted"]),
            "actual": pretty(r["actual"]),
            "hit": bool(r["hit"]),
            "confidence": round(float(r["confidence"]), 2),
        } for idx, r in enumerate(rows)]

        timeline = self._build_timeline(rows)
        drift_events = self._detect_drift_events(rows)

        # Heatmap is already calculated above

        return {
            "accuracy": accuracy,
            "lastSynced": now_str,
            "timeline": timeline,
            "decisions": decisions,
            "driftEvents": drift_events,
            "data_maturity": maturity,
            "heatmap": heatmap,
        }

    def _build_timeline(self, rows: list[dict]) -> list[dict]:
        """Aggregate per-day: actual line at 1.0, predicted = P(actual), confidence."""
        by_date: dict[str, list[dict]] = {}
        for r in rows:
            date = r["timestamp"].strftime("%Y-%m-%d")
            by_date.setdefault(date, []).append(r)
        timeline = []
        for date in sorted(by_date):
            day = by_date[date]
            predicted = sum(x["prob_actual"] for x in day) / len(day)
            confidence = sum(x["confidence"] for x in day) / len(day)
            timeline.append({
                "date": date,
                "actual": 1,
                "predicted": round(predicted, 3),
                "confidence": round(confidence, 3),
            })
        return timeline

    def _detect_drift_events(self, rows: list[dict]) -> list[dict]:
        """Surface plain-language drift moments per domain from rolling accuracy."""
        events = []
        for domain in self.domains:
            dom_rows = [r for r in rows if r["domain"] == domain]
            if len(dom_rows) < _DRIFT_WINDOW:
                continue
            in_drift = False
            for i in range(_DRIFT_WINDOW, len(dom_rows) + 1):
                window = dom_rows[i - _DRIFT_WINDOW:i]
                acc = sum(1 for w in window if w["hit"]) / len(window)
                if acc < _DRIFT_THRESHOLD and not in_drift:
                    in_drift = True
                    weekend_miss = sum(1 for w in window
                                       if not w["hit"] and w["day_type"] == "weekend")
                    misses = sum(1 for w in window if not w["hit"])
                    note = (f"{domain.capitalize()} predictions became less reliable")
                    if misses and weekend_miss / max(misses, 1) > 0.5:
                        note += " on weekends."
                    else:
                        note += f" (accuracy dropped to {int(acc * 100)}%)."
                    events.append({
                        "date": window[-1]["timestamp"].strftime("%Y-%m-%d"),
                        "domain": domain, "note": note,
                    })
                elif acc >= _DRIFT_THRESHOLD:
                    in_drift = False
        events.sort(key=lambda e: e["date"])
        return events

    # ------------------------------------------------------------------
    def predict_next(self, user_id: str, domain: str = "focus") -> dict:
        """Predict the user's next decision for ``domain`` from recent history."""
        self._ensure_trained(user_id)
        if domain not in self._models.get(user_id, {}):
            raise KeyError(domain)
        dm = self._models[user_id][domain]
        records = self.store.load(user_id=user_id)
        dom_records = [r for r in records if r.domain == domain]
        recent = dom_records[-dm.pipe.k:]
        last = dom_records[-1] if dom_records else None

        class _Ctx:
            location = last.location if last else "home"
            weather = last.weather if last else "clear"
            day_type = last.day_type if last else "weekday"
            time_of_day = last.time_of_day if last else "morning"
            mood_energy = last.mood_energy if last else 0.5
            stress_level = getattr(last, "stress_level", "medium") if last else "medium"
            timestamp = datetime.now()

        t0 = time.perf_counter()
        x, seq = dm.pipe.transform_one(_Ctx(), recent, self.profiles.get(user_id))
        probs = dm.model.predict_proba(x, seq)
        latency_ms = (time.perf_counter() - t0) * 1000

        predicted = max(probs, key=probs.get)
        confidence = float(probs[predicted])

        result = {
            "user_id": user_id, "domain": domain,
            "predicted_decision": pretty(predicted),
            "confidence": round(confidence, 3),
            "class_probs": {pretty(k): round(float(v), 3) for k, v in probs.items()},
            "model_name": dm.model_name,
        }

        # MLOps: log prediction (Req 18.1, 18.9 — failures MUST NOT block response)
        try:
            if self._prediction_logger is not None:
                model_version = dm.metrics.get("version", dm.model_name or "unknown")
                self._prediction_logger.log(
                    user_id=user_id,
                    domain=domain,
                    prediction=predicted,
                    confidence=confidence,
                    model_version=str(model_version),
                    latency_ms=latency_ms,
                    timestamp=datetime.now(timezone.utc),
                )
        except Exception as exc:
            _svc_logger.error("prediction_log_failed user_id=%s domain=%s exc=%s",
                              user_id, domain, exc)

        return result

    def predict_next_existing(self, user_id: str, domain: str = "focus") -> dict:
        """Predict the next decision for ``domain`` using ONLY an already-trained model.

        Unlike :meth:`predict_next`, this NEVER auto-trains or seeds synthetic data: if no
        model artifact exists for ``(user_id, domain)`` it raises
        :class:`ModelNotTrainedError` so the Phase 3 ``GET /predict_next_decision/{id}``
        path can return HTTP 409 *without mutating any stored state* (Requirement 14.4).

        ``predicted_decision`` is the RAW option value drawn from the domain Option_Set
        (Requirement 14.3); ``confidence`` is the maximum class probability in ``[0, 1]``.
        """
        dm = self._models.get(user_id, {}).get(domain)
        if dm is None:
            raise ModelNotTrainedError(domain)
        # Reads only (no append/seed/train), so stored state is left untouched.
        records = self.store.load(user_id=user_id)
        dom_records = [r for r in records if r.domain == domain]
        recent = dom_records[-dm.pipe.k:]
        last = dom_records[-1] if dom_records else None

        class _Ctx:
            location = last.location if last else "home"
            weather = last.weather if last else "clear"
            day_type = last.day_type if last else "weekday"
            time_of_day = last.time_of_day if last else "morning"
            mood_energy = last.mood_energy if last else 0.5
            stress_level = getattr(last, "stress_level", "medium") if last else "medium"
            timestamp = datetime.now()

        x, seq = dm.pipe.transform_one(_Ctx(), recent, self.profiles.get(user_id))
        probs = dm.model.predict_proba(x, seq)
        predicted = max(probs, key=probs.get)
        return {
            "user_id": user_id, "domain": domain,
            "predicted_decision": predicted,
            "confidence": round(float(probs[predicted]), 3),
        }

    def history(self, user_id: str) -> dict:
        """Return accuracy, lastSynced, timeline, and decisions (no drift events)."""
        full = self.dashboard(user_id)
        return {k: full[k] for k in ("accuracy", "lastSynced", "timeline", "decisions")}

    def drift_events(self, user_id: str) -> list[dict]:
        return self.dashboard(user_id)["driftEvents"]

    def user_profile(self, user_id: str) -> dict:
        self._ensure_trained(user_id)
        p = self.profiles.get(user_id)
        return {
            "user_id": user_id,
            "decision_counts": p.decision_counts,
            "embedding_summary": p.embedding_summary(),
            "last_updated": _iso_z(p.last_updated) if p.last_updated else None,
        }

    def retrain(self, user_id: str) -> dict:
        """Retrain on the latest stored data; return status + metrics per domain."""
        records = self.store.load(user_id=user_id)
        dom_counts = {d: sum(1 for r in records if r.domain == d) for d in self.domains}
        if all(c < 5 for c in dom_counts.values()):
            return {"status": "skipped", "reason": "insufficient_data",
                    "lastSynced": _iso_z(datetime.now(timezone.utc))}
        reports = self.train(user_id)
        return {"status": "retrained", "metrics": reports,
                "lastSynced": _iso_z(datetime.now(timezone.utc))}

    # ------------------------------------------------------------------
    # Requirement 7: model-serving core (POST /predict_next_decision, POST /retrain)
    # ------------------------------------------------------------------
    def _records_from_labels(self, domain: str, labels: list[str], ctx) -> list:
        """Build synthetic, time-ordered DecisionRecords from recent decision labels.

        Only ``domain``, ``decision_made``, and ``timestamp`` are consumed by
        :meth:`FeaturePipeline.transform_one`; the remaining fields are filled from the
        request context so the records are well-formed. Unseen labels are handled
        downstream by the history encoder (they map to the ``<UNK>`` bucket).
        """
        base = datetime.now()
        recs = []
        n = len(labels)
        for i, label in enumerate(labels):
            recs.append(DecisionRecord(
                user_id="", timestamp=base - timedelta(minutes=(n - i)),
                domain=domain, location=getattr(ctx, "location", "home"),
                weather=getattr(ctx, "weather", "clear"),
                day_type=getattr(ctx, "day_type", "weekday"),
                time_of_day=getattr(ctx, "time_of_day", "morning"),
                mood_energy=float(getattr(ctx, "mood_energy", 0.5)),
                stress_level=getattr(ctx, "stress_level", "medium"),
                decision_made=label, outcome=""))
        return recs

    def _drift_status_for(self, user_id: str, domain: str) -> dict:
        """Current drift status for ``domain`` from the held-out walk-forward tail.

        Replays the validation predictions through a :class:`DriftDetector` so the
        returned status reflects recent rolling accuracy (Component 6 / Requirement 7.1).
        """
        det = DriftDetector(window=_DRIFT_WINDOW, threshold=_DRIFT_THRESHOLD)
        for r in self._walk_forward(user_id):
            if r["domain"] == domain:
                det.record(r["predicted"], r["actual"], r["confidence"])
        return det.status().as_dict()

    def predict_next_raw(self, user_id: str, context, recent_decisions: list[str]) -> dict:
        """Predict the next decision for ``context.domain`` (Requirement 7.1-7.3, 7.6).

        Unlike :meth:`predict_next` this does NOT auto-train: if no model artifact
        exists for the requested domain it raises :class:`ModelNotTrainedError` so the
        API can return HTTP 409 directing the client to ``/retrain`` (Requirement 7.6).

        Returns ``predicted_decision`` as the raw option value (in the domain
        ``Option_Set``), ``class_probs`` keyed by raw options (a valid distribution),
        ``confidence`` equal to ``max(class_probs)``, plus ``drift_status`` and
        ``model_name``.
        """
        domain = getattr(context, "domain", None)
        if domain not in self.domains:
            raise ModelNotTrainedError(str(domain))
        dm = self._models.get(user_id, {}).get(domain)
        if dm is None:
            raise ModelNotTrainedError(domain)

        t0 = time.perf_counter()
        recent = self._records_from_labels(domain, recent_decisions, context)
        x, seq = dm.pipe.transform_one(context, recent, self.profiles.get(user_id))
        raw_probs = dm.model.predict_proba(x, seq)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Guarantee full option coverage + a valid normalized distribution so the
        # response invariants (non-negative, sum~1, confidence==max) always hold.
        class_probs = build_class_probs(list(options(domain)), raw_probs)
        predicted = max(class_probs, key=class_probs.get)
        confidence = class_probs[predicted]

        result = {
            "predicted_decision": predicted,
            "confidence": float(confidence),
            "class_probs": {k: float(v) for k, v in class_probs.items()},
            "drift_status": self._drift_status_for(user_id, domain),
            "model_name": dm.model_name,
        }

        # MLOps: log prediction (Req 18.1, 18.9 — failures MUST NOT block response)
        try:
            if self._prediction_logger is not None:
                model_version = dm.metrics.get("version", dm.model_name or "unknown")
                self._prediction_logger.log(
                    user_id=user_id,
                    domain=domain,
                    prediction=predicted,
                    confidence=float(confidence),
                    model_version=str(model_version),
                    latency_ms=latency_ms,
                    timestamp=datetime.now(timezone.utc),
                )
        except Exception as exc:
            _svc_logger.error("prediction_log_failed user_id=%s domain=%s exc=%s",
                              user_id, domain, exc)

        # MLOps: append drift status (Req 18.2 — failures MUST NOT block response)
        try:
            if self._drift_monitor is not None:
                drift_s = result["drift_status"]
                win_acc = drift_s.get("window_acc") or 0.0
                drift_flag = bool(drift_s.get("drift", False))
                self._drift_monitor.append(
                    user_id=user_id,
                    domain=domain,
                    window_acc=float(win_acc),
                    drift=drift_flag,
                    timestamp=datetime.now(timezone.utc),
                )
        except Exception as exc:
            _svc_logger.error("drift_monitor_failed user_id=%s domain=%s exc=%s",
                              user_id, domain, exc)

        return result

    def retrain_models(self, user_id: str | None) -> dict:
        """Retrain per domain and return status + per-domain winner metrics (Req 7.5, 7.7).

        When every domain has too few records for a valid temporal split the previously
        trained artifacts are left untouched and ``{"status": "skipped",
        "reason": "insufficient_data"}`` is returned (Requirement 7.7). Otherwise the
        winning model per domain is retrained and its evaluation metrics are returned.
        """
        target = user_id or "demo_user"
        records = self.store.load(user_id=target)
        dom_counts = {d: sum(1 for r in records if r.domain == d) for d in self.domains}
        # Too few records anywhere for a valid temporal split -> keep prior artifacts.
        if all(c < 5 for c in dom_counts.values()):
            return {"status": "skipped", "reason": "insufficient_data"}

        reports = self.train(target)
        # Return the FULL comparison report per domain so the frontend
        # can display baseline vs sequence model comparison (MLOps panel).
        metrics: dict[str, dict] = {}
        for rep in reports:
            metrics[rep["domain"]] = rep  # full dict: baseline, sequence, winner, rationale

        # MLOps: record retrain per domain (Req 18.4)
        try:
            if self._retrain_trigger is not None:
                for domain in self.domains:
                    self._retrain_trigger.record_retrain(domain)
        except Exception as exc:
            _svc_logger.error("retrain_trigger_record_failed exc=%s", exc)

        return {"status": "retrained", "metrics": metrics}

    # ------------------------------------------------------------------
    def reset_user(self, user_id: str) -> None:
        """Completely wipe a user's data from DB, profiles, and memory."""
        # 1. Delete from DB (SQLite specifically)
        if self.store.backend == "sqlite":
            with self.store._connect() as conn:
                conn.execute("DELETE FROM decisions WHERE user_id = ?", (user_id,))
                
        # 2. Delete Personalization Profile
        if hasattr(self.profiles, 'delete'):
            self.profiles.delete(user_id)
            
        # 3. Clear In-Memory Model Cache
        if user_id in self._models:
            del self._models[user_id]
        if user_id in self._reports:
            del self._reports[user_id]

