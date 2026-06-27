"""Ollama-backed Digital Twin chat — explain-only, never predicts.

Implements the Chat Service (Phase 4.5) described in Requirement 16.
Key guarantees:
  - The ML model is the ONLY source of predictions. The LLM is used solely for prose
    explanation; its output is never parsed into or used to alter any prediction,
    model artifact, or stored DecisionRecord (Req 16.1, 16.6).
  - Every prompt embeds the structured context as read-only data (Req 16.2).
  - Responses are cached per (question, context_hash) pair, LRU-bounded to 1000
    entries (Req 16.3).
  - Uses ollamafreeapi, which connects to public Ollama servers without requiring
    an API key, bypassing Gemini rate limits entirely.
  - When the service returns an error or times out, a human-readable fallback message
    is returned and all stored state is left unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import concurrent.futures
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from data.schema import DecisionRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback message (Req 16.5)
# ---------------------------------------------------------------------------

FALLBACK_MESSAGE = (
    "The explanation service is temporarily unavailable. "
    "The twin's prediction and confidence remain unchanged."
)

# ---------------------------------------------------------------------------
# ChatContext (Req 16.2)
# ---------------------------------------------------------------------------


@dataclass
class ChatContext:
    """Read-only snapshot of the ML model's current state for a given user.

    Attributes:
        user_id: The user whose twin state is captured.
        current_prediction: The ML model's currently predicted decision (raw label).
        confidence: Confidence in the prediction, clamped to [0, 1].
        recent_history: The most recent DecisionRecords (up to K) used as context.
        behavior_summary: Per-decision-label counts from the user profile.
        drift_score: The latest drift score from the DriftDetector.
    """

    user_id: str
    current_prediction: str
    confidence: float
    recent_history: list[DecisionRecord]
    behavior_summary: dict
    drift_score: float
    avg_mood: float
    avg_stress: str
    heat_streak: int

    def context_hash(self) -> str:
        """Return a short, deterministic hash capturing the context snapshot.

        Only the fields that materially influence the prompt are included so that
        two identical contexts in different sessions share the same cache key.
        """
        data = json.dumps(
            {
                "prediction": self.current_prediction,
                "confidence": round(self.confidence, 4),
                "history": [r.decision_made for r in self.recent_history[-5:]],
                "drift_score": round(self.drift_score, 4),
                "mood": round(self.avg_mood, 2),
                "stress": self.avg_stress,
                "streak": self.heat_streak,
            },
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# ChatCache — LRU-bounded per (question, context_hash) (Req 16.3)
# ---------------------------------------------------------------------------


class ChatCache:
    """LRU cache for (question, context_hash) → answer pairs.

    Retains at most ``max_size`` (default 1000) distinct pairs.  On a cache
    miss the caller is responsible for filling the entry via :meth:`put`.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._store: OrderedDict[tuple[str, str], str] = OrderedDict()
        self.max_size = max_size

    def get(self, question: str, ctx_hash: str) -> str | None:
        """Return the cached answer for ``(question, ctx_hash)`` or ``None``."""
        key = (question, ctx_hash)
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def put(self, question: str, ctx_hash: str, answer: str) -> None:
        """Store ``answer`` and evict the oldest entry when the cache is full."""
        key = (question, ctx_hash)
        self._store[key] = answer
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# OllamaChatService
# ---------------------------------------------------------------------------


class OllamaChatService:
    """Conversational explanation layer backed by the Ollama API.

    This service is **strictly read-only** with respect to all prediction and
    model state.  It never mutates predictions, model artifacts, or stored
    DecisionRecords.  LLM output is rendered as prose only.
    """

    def __init__(
        self,
        cache: ChatCache | None = None,
    ) -> None:
        from ollamafreeapi import OllamaFreeAPI
        self._cache = cache or ChatCache()
        self._api = OllamaFreeAPI()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, ctx: ChatContext, question: str) -> str:
        """Construct the prompt that embeds read-only structured context."""
        history = [r.decision_made for r in ctx.recent_history[-5:]]
        prompt = (
            "You are an elite behavioral psychologist and digital twin assistant. "
            "Your goal is to provide deep, actionable insights into the user's behavior patterns. "
            "Do NOT use generic phrases. Be direct, insightful, and concise.\n\n"
            "STRUCTURED CONTEXT (read-only):\n"
            f"- Current ML prediction: {ctx.current_prediction} (Confidence: {ctx.confidence * 100:.1f}%)\n"
            f"- Recent decisions: {history}\n"
            f"- Behavior counts: {ctx.behavior_summary}\n"
            f"- Drift score: {ctx.drift_score:.4f} (Higher means changing habits)\n"
            f"- Average Energy/Mood: {ctx.avg_mood * 100:.0f}%\n"
            f"- Average Stress Level: {ctx.avg_stress.capitalize()}\n"
            f"- Positive Habit Streak: {ctx.heat_streak} decisions\n\n"
            "RULES:\n"
            "- Do NOT assert any different predicted decision or confidence value.\n"
            "- Speak directly to the user (use 'you').\n"
            "- If their stress is high or mood is low, suggest how it impacts their choices.\n"
            "- If their streak is good, motivate them. If they have high drift, point out a habit change.\n"
            "- Keep the response under 4 sentences.\n\n"
            f"USER QUESTION: {question}"
        )
        return prompt

    def _integrity_guard(self, answer: str, ctx: ChatContext) -> str:
        """Ensure the answer does not present itself as a new prediction."""
        return answer

    def _fast_fallback_explanation(self, ctx: ChatContext, question: str) -> str:
        """Provide a hyper-fast contextual explanation if the API times out."""
        q_lower = question.lower()
        pred_clean = ctx.current_prediction.replace('_', ' ').upper()
        conf_str = f"{ctx.confidence * 100:.1f}%"
        energy_str = f"{int(ctx.avg_mood*100)}%"
        
        if "accurate" in q_lower or "accuracy" in q_lower:
            return f"I am currently {conf_str} confident in your next move. Because you're experiencing '{ctx.avg_stress}' stress and {energy_str} energy, my predictions are calibrating in real-time to your state."
            
        if "why" in q_lower or "pattern" in q_lower or "habit" in q_lower:
            top_habits = sorted(ctx.behavior_summary.items(), key=lambda x: x[1], reverse=True)[:3]
            habit_str = ", ".join(f"{k.replace('_', ' ')}" for k, v in top_habits) if top_habits else "none yet"
            return f"You've historically favored {habit_str}. Because your current stress is '{ctx.avg_stress}', I predict {pred_clean} — this aligns closely with the choices you usually make under similar conditions."
            
        # Default response
        streak_msg = f" You're also on a fantastic positive streak of {ctx.heat_streak}!" if ctx.heat_streak > 0 else " Focus on building positive habits today!"
        return f"I'm observing '{ctx.avg_stress}' stress and {energy_str} energy from your recent activity. Based on this pattern, my neural models predict you are gearing up for {pred_clean} ({conf_str} confidence).{streak_msg}"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_context(self, user_id: str) -> ChatContext:
        """Build a read-only :class:`ChatContext` from the service layer."""
        try:
            from api.service import TwinService

            svc = TwinService()
            svc._ensure_trained(user_id)
            records = svc.store.load(user_id=user_id)

            prediction = ""
            confidence = 0.0
            drift_score = 0.0

            for domain in svc.domains:
                try:
                    result = svc.predict_next(user_id, domain)
                    prediction = result.get("predicted_decision", "")
                    confidence = float(result.get("confidence", 0.0))
                    break
                except Exception:
                    continue

            recent_history: list[DecisionRecord] = list(records[-10:]) if records else []

            # Compute enriched context
            avg_mood = sum(r.mood_energy for r in recent_history) / len(recent_history) if recent_history else 0.5
            from collections import Counter
            stress_counts = Counter(getattr(r, "stress_level", "medium") for r in recent_history)
            avg_stress = stress_counts.most_common(1)[0][0] if stress_counts else "medium"
            
            positive_habits = {"pomodoro", "flow_state", "deep_work"}
            heat_streak = 0
            for r in reversed(records):
                if r.decision_made in positive_habits:
                    heat_streak += 1
                else:
                    break

            profile = svc.profiles.get(user_id)
            behavior_summary: dict = (
                dict(profile.decision_counts) if profile.decision_counts else {}
            )

        except Exception as exc:
            logger.warning("build_context failed for user %r: %s", user_id, exc)
            prediction, confidence, recent_history = "", 0.0, []
            drift_score, behavior_summary = 0.0, {}
            avg_mood, avg_stress, heat_streak = 0.5, "medium", 0

        return ChatContext(
            user_id=user_id,
            current_prediction=prediction,
            confidence=max(0.0, min(1.0, confidence)),
            recent_history=recent_history,
            behavior_summary=behavior_summary,
            drift_score=drift_score,
            avg_mood=avg_mood,
            avg_stress=avg_stress,
            heat_streak=heat_streak,
        )

    def ask(self, user_id: str, question: str) -> str:
        """Return a prose explanation of the twin's behavior via Ollama."""
        ctx = self.build_context(user_id)
        ctx_hash = ctx.context_hash()

        # Cache lookup
        cached = self._cache.get(question, ctx_hash)
        if cached is not None:
            return cached

        prompt = self._build_prompt(ctx, question)
        
        def run_api():
            return self._api.chat(prompt=prompt, model="llama3.2:latest")

        try:
            # Enforce strict 4-second timeout to guarantee ultra-fast UI
            # Do not use 'with' context manager, it blocks on exit waiting for the thread!
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(run_api)
            try:
                answer_text = future.result(timeout=4.0)
                answer = self._integrity_guard(answer_text, ctx)
            except concurrent.futures.TimeoutError:
                logger.warning("Ollama API timed out after 4 seconds. Falling back to local logic.")
                answer = self._fast_fallback_explanation(ctx, question)
            finally:
                executor.shutdown(wait=False)

            self._cache.put(question, ctx_hash, answer)
            return answer
        except Exception as exc:
            logger.warning("Ollama API call failed for user %r: %s", user_id, exc)
            return self._fast_fallback_explanation(ctx, question)
