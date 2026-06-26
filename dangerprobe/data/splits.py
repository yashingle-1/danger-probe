"""Load the dataset and produce train/test splits.

Two split modes (config.split.mode):
    random                 : stratified random split by label
    leave_one_category_out : train on all dangerous framings EXCEPT
                             config.split.holdout_category; test on the held-out
                             framing (+ a benign slice). This is the
                             generalization eval — does the probe catch harm
                             under a framing it never trained on?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dangerprobe.config import Config


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found: {path}. Run `python -m dangerprobe.data.build_dataset` first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _stratified_random(labels: np.ndarray, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(test_idx)


def _leave_one_category_out(records: list[dict], holdout: str, test_frac: float, seed: int):
    """Test = held-out dangerous framing + a benign slice; train = the rest."""
    rng = np.random.default_rng(seed)
    cats = np.array([r["threat_category"] for r in records])
    labels = np.array([r["label"] for r in records])

    held_dangerous = np.where((cats == holdout) & (labels == 1))[0]
    if len(held_dangerous) == 0:
        raise ValueError(
            f"holdout_category '{holdout}' has no dangerous examples; "
            f"present categories: {sorted(set(cats.tolist()))}"
        )

    benign_idx = np.where(labels == 0)[0]
    rng.shuffle(benign_idx)
    n_benign_test = int(round(len(benign_idx) * test_frac))
    benign_test = benign_idx[:n_benign_test]
    benign_train = benign_idx[n_benign_test:]

    test_idx = np.concatenate([held_dangerous, benign_test])
    train_dangerous = np.where((cats != holdout) & (labels == 1))[0]
    train_idx = np.concatenate([train_dangerous, benign_train])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def make_split(records: list[dict], cfg: Config):
    """Return (train_idx, test_idx) as numpy arrays per config.split."""
    sp = cfg.split
    labels = np.array([r["label"] for r in records])
    mode = sp["mode"]
    if mode == "random":
        return _stratified_random(labels, sp["test_frac"], cfg.seed)
    if mode == "leave_one_category_out":
        return _leave_one_category_out(
            records, sp["holdout_category"], sp["test_frac"], cfg.seed
        )
    raise ValueError(f"unknown split mode: {mode}")
