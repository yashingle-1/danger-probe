"""Trained probe: scaler + linear classifier bound to one (layer, pool).

Shared by the trainer (which fits and saves the best one) and the monitor
(which loads it and scores live activations).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Probe:
    clf: Any              # fitted sklearn classifier with predict_proba
    scaler: Any           # fitted StandardScaler (or None)
    layer: int
    pool: str             # "last" | "mean"
    model_name: str
    safe_threshold: float = 0.3
    block_threshold: float = 0.7

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def proba(self, X: np.ndarray) -> np.ndarray:
        """P(dangerous) for each row of X."""
        return self.clf.predict_proba(self._prep(X))[:, 1]

    def proba_one(self, vec: np.ndarray) -> float:
        return float(self.proba(vec)[0])

    def decide(self, p: float) -> str:
        if p < self.safe_threshold:
            return "SAFE"
        if p > self.block_threshold:
            return "BLOCK"
        return "WARN"

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Probe":
        import joblib

        return joblib.load(path)
