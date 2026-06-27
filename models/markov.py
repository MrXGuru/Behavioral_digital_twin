"""First-Order Markov Chain baseline for sequence modeling.

Learns the transition probabilities from the immediate previous decision (the last
element in the sequence window) to the next decision. If the sequence is empty (K=0)
or an unobserved transition is requested, it falls back to the prior probabilities
of the training labels.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from models.base import DecisionModel, validate_distribution


class MarkovModel(DecisionModel):
    """First-Order Markov Chain baseline."""

    name = "markov"

    def __init__(self, options: list[str]) -> None:
        super().__init__(options)
        self._prior_probs: dict[str, float] = {}
        # transition_probs[prev_decision][next_decision] = float
        self._transition_probs: dict[str, dict[str, float]] = defaultdict(dict)

    def fit(self, X: np.ndarray, seq: np.ndarray, y: list[str]) -> "MarkovModel":
        # Calculate priors
        priors = {opt: 0.0 for opt in self.options}
        for label in y:
            if label in priors:
                priors[label] += 1.0
        total_priors = sum(priors.values()) or 1.0
        self._prior_probs = {k: v / total_priors for k, v in priors.items()}

        # If sequence is not provided or K=0, we just use priors
        if seq is None or seq.shape[1] == 0:
            return self

        # Calculate transitions
        transitions = defaultdict(lambda: {opt: 0.0 for opt in self.options})
        for i, label in enumerate(y):
            # The immediate previous decision is the last step in the sequence
            # seq shape is (N, K, vocab_size). The last step is seq[i, -1, :]
            last_step = seq[i, -1, :]
            # It's one-hot encoded. Find the index where it is 1.0
            idx = np.argmax(last_step)
            # Check if it's actually one-hot (could be all zeros if padded)
            if last_step[idx] > 0.5:
                # The index corresponds to the option in the label_space
                try:
                    prev_decision = self.label_space.decode(idx)
                    transitions[prev_decision][label] += 1.0
                except IndexError:
                    pass

        # Normalize transitions
        for prev_decision, counts in transitions.items():
            total = sum(counts.values())
            if total > 0:
                self._transition_probs[prev_decision] = {
                    k: v / total for k, v in counts.items()
                }

        return self

    def predict_proba(self, x: np.ndarray, seq: np.ndarray) -> dict[str, float]:
        probs = self._prior_probs.copy()
        
        if seq is not None and seq.shape[0] > 0:
            # We assume seq is (K, vocab_size) or (1, K, vocab_size) 
            # In evaluate / predict paths, it is (1, K, vocab_size) if passed individually 
            # But the signature implies x, seq are for one sample (or flattened).
            # In _score_model, seq is sliced as fm_val.seq[i:i+1] which is (1, K, vocab_size)
            if len(seq.shape) == 3:
                last_step = seq[0, -1, :]
            elif len(seq.shape) == 2:
                last_step = seq[-1, :]
            else:
                last_step = seq
                
            idx = np.argmax(last_step)
            if last_step[idx] > 0.5:
                try:
                    prev_decision = self.label_space.decode(idx)
                    if prev_decision in self._transition_probs:
                        probs = self._transition_probs[prev_decision].copy()
                except IndexError:
                    pass

        # Ensure all options exist and validate
        for opt in self.options:
            probs.setdefault(opt, 0.0)
        
        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}
        validate_distribution(probs)
        return probs

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "options": self.options,
                "prior_probs": self._prior_probs,
                "transition_probs": dict(self._transition_probs),
            }, fh)

    @classmethod
    def load(cls, path: str) -> "MarkovModel":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        model = cls(state["options"])
        model._prior_probs = state["prior_probs"]
        model._transition_probs = defaultdict(dict, state["transition_probs"])
        return model
