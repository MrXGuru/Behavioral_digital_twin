"""Sequence twin: a small LSTM over the last-K decision window plus context.

Uses PyTorch when available (a single-layer LSTM kept small enough to train on CPU). When
torch is not installed, it transparently falls back to a scikit-learn ``MLPClassifier``
over the flattened sequence concatenated with the flat feature vector, so the repo runs
and the model comparison is meaningful in any environment. Both paths expose the same
``DecisionModel`` interface and a save/load round-trip.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from models.base import DecisionModel, validate_distribution

try:  # optional heavy dependency
    import torch
    import torch.nn as nn

    _TORCH = True
except Exception:  # pragma: no cover - exercised only when torch missing
    _TORCH = False


class _LSTMNet:  # thin wrapper so type checkers don't choke when torch is absent
    pass


if _TORCH:

    class LSTMClassifier(nn.Module):
        """Multi-layer LSTM with a context-augmented classification head."""

        def __init__(self, step_dim: int, ctx_dim: int, n_classes: int, hidden: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(step_dim, hidden, num_layers=2, dropout=0.2, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden + ctx_dim, hidden), 
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, n_classes),
            )

        def forward(self, seq, ctx):
            _, (h, _) = self.lstm(seq)
            feat = torch.cat([h[-1], ctx], dim=1)
            return self.head(feat)


class SequenceModel(DecisionModel):
    """LSTM (torch) or MLP-fallback sequence twin for one domain."""

    name = "sequence"

    def __init__(self, options: list[str], epochs: int = 15, hidden: int = 16) -> None:
        super().__init__(options)
        self.epochs = epochs
        self.hidden = hidden
        self.backend = "torch" if _TORCH else "sklearn"
        self._net = None
        self._mlp = None
        self._classes: list[str] = []
        self._fallback_probs: dict[str, float] | None = None
        self._step_dim = 0
        self._ctx_dim = 0

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, seq: np.ndarray, y: list[str]) -> "SequenceModel":
        self._classes = sorted(set(y))
        if len(self._classes) < 2 or len(y) < 10:
            self._fallback_probs = self._prior_from_labels(y)
            return self

        if self.backend == "torch":
            self._fit_torch(X, seq, y)
        else:
            self._fit_sklearn(X, seq, y)
        return self

    def _prior_from_labels(self, y: list[str]) -> dict[str, float]:
        probs = {opt: 0.0 for opt in self.options}
        for label in y:
            if label in probs:
                probs[label] += 1.0
        total = sum(probs.values()) or 1.0
        return {k: v / total for k, v in probs.items()}

    # -- torch path --------------------------------------------------------
    def _fit_torch(self, X, seq, y):
        self._step_dim = seq.shape[2]
        self._ctx_dim = X.shape[1]
        cls_to_idx = {c: i for i, c in enumerate(self._classes)}
        y_idx = torch.tensor([cls_to_idx[v] for v in y], dtype=torch.long)
        seq_t = torch.tensor(seq, dtype=torch.float32)
        ctx_t = torch.tensor(X, dtype=torch.float32)

        self._net = LSTMClassifier(self._step_dim, self._ctx_dim, len(self._classes),
                                   self.hidden)
        opt = torch.optim.Adam(self._net.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        self._net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logits = self._net(seq_t, ctx_t)
            loss = loss_fn(logits, y_idx)
            loss.backward()
            opt.step()
        self._net.eval()

    # -- sklearn fallback --------------------------------------------------
    def _flatten(self, X, seq):
        n = seq.shape[0]
        return np.concatenate([seq.reshape(n, -1), X], axis=1)

    def _fit_sklearn(self, X, seq, y):
        from sklearn.neural_network import MLPClassifier

        feats = self._flatten(X, seq)
        self._mlp = MLPClassifier(hidden_layer_sizes=(self.hidden,), max_iter=10,
                                  random_state=0)
        try:
            self._mlp.fit(feats, y)
        except Exception:
            self._mlp = None
            self._fallback_probs = self._prior_from_labels(y)

    # ------------------------------------------------------------------
    def predict_proba(self, x: np.ndarray, seq: np.ndarray) -> dict[str, float]:
        if self._fallback_probs is not None and self._net is None and self._mlp is None:
            probs = dict(self._fallback_probs)
        elif self.backend == "torch" and self._net is not None:
            with torch.no_grad():
                logits = self._net(torch.tensor(seq, dtype=torch.float32),
                                   torch.tensor(x, dtype=torch.float32))
                row = torch.softmax(logits, dim=1)[0].numpy()
            probs = {c: float(p) for c, p in zip(self._classes, row)}
        elif self._mlp is not None:
            feats = self._flatten(x, seq)
            row = self._mlp.predict_proba(feats)[0]
            probs = {c: float(p) for c, p in zip(self._mlp.classes_, row)}
        else:
            probs = {opt: 1.0 / len(self.options) for opt in self.options}

        for opt in self.options:
            probs.setdefault(opt, 0.0)
        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}
        validate_distribution(probs)
        return probs

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "options": self.options, "epochs": self.epochs, "hidden": self.hidden,
            "backend": self.backend, "classes": self._classes,
            "fallback": self._fallback_probs, "step_dim": self._step_dim,
            "ctx_dim": self._ctx_dim, "mlp": self._mlp,
            "torch_state": (self._net.state_dict() if (_TORCH and self._net) else None),
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

    @classmethod
    def load(cls, path: str) -> "SequenceModel":
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        model = cls(state["options"], epochs=state["epochs"], hidden=state["hidden"])
        model.backend = state["backend"]
        model._classes = state["classes"]
        model._fallback_probs = state["fallback"]
        model._step_dim = state["step_dim"]
        model._ctx_dim = state["ctx_dim"]
        model._mlp = state["mlp"]
        if _TORCH and state["torch_state"] is not None:
            model._net = LSTMClassifier(model._step_dim, model._ctx_dim,
                                        len(model._classes), model.hidden)
            model._net.load_state_dict(state["torch_state"])
            model._net.eval()
        return model
