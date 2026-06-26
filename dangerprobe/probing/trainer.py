"""Train a probe per (layer, pooling), pick the best, save it.

Reads saved activations, applies the configured train/test split, fits a
logistic-regression (or MLP) probe on each (layer, pool), reports a metrics
table, and saves the best probe by test F1 to config.paths.models_dir.

CLI:
    python -m dangerprobe.probing.trainer
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from dangerprobe.config import Config, load_config
from dangerprobe.data.splits import make_split
from dangerprobe.probing.extractor import model_slug
from dangerprobe.probing.probe import Probe


@dataclass
class ProbeResult:
    layer: int
    pool: str
    accuracy: float
    f1: float
    precision: float
    recall: float
    auc: float
    fpr: float  # false-positive rate on benign at 0.5


def _metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (p >= 0.5).astype(int)
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "auc": auc,
        "fpr": fpr,
    }


def _make_clf(cfg: Config):
    p = cfg.raw["probe"]
    if p["type"] == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=p["C"], max_iter=p["max_iter"])
    if p["type"] == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(hidden_layer_sizes=(256,), max_iter=p["max_iter"])
    raise ValueError(f"unknown probe type: {p['type']}")


def _load_meta_records(act_dir: Path) -> list[dict]:
    with open(act_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    labels = np.load(act_dir / "labels.npy")
    return [
        {"label": int(lbl), "threat_category": cat}
        for lbl, cat in zip(labels, meta["categories"])
    ]


def train(cfg: Config) -> tuple[list[ProbeResult], Probe]:
    from sklearn.preprocessing import StandardScaler

    mc = cfg.model_config()
    act_dir = cfg.path("activations_dir") / model_slug(mc.name)
    if not (act_dir / "labels.npy").exists():
        raise FileNotFoundError(
            f"no activations at {act_dir}. Run `python -m dangerprobe.probing.extractor` first."
        )

    labels = np.load(act_dir / "labels.npy")
    records = _load_meta_records(act_dir)
    train_idx, test_idx = make_split(records, cfg)
    standardize = cfg.raw["probe"]["standardize"]

    results: list[ProbeResult] = []
    best: tuple[float, Probe] | None = None

    for L in mc.layers:
        for pool in cfg.pooling_methods:
            arr = np.load(act_dir / f"layer{L}_{pool}.npy")
            Xtr, Xte = arr[train_idx], arr[test_idx]
            ytr, yte = labels[train_idx], labels[test_idx]

            scaler = StandardScaler().fit(Xtr) if standardize else None
            Xtr_s = scaler.transform(Xtr) if scaler else Xtr
            Xte_s = scaler.transform(Xte) if scaler else Xte

            clf = _make_clf(cfg)
            clf.fit(Xtr_s, ytr)
            p = clf.predict_proba(Xte_s)[:, 1]
            m = _metrics(yte, p)
            results.append(ProbeResult(layer=L, pool=pool, **m))

            if best is None or m["f1"] > best[0]:
                th = cfg.thresholds
                probe = Probe(
                    clf=clf,
                    scaler=scaler,
                    layer=L,
                    pool=pool,
                    model_name=mc.name,
                    safe_threshold=th["safe"],
                    block_threshold=th["block"],
                )
                best = (m["f1"], probe)

    assert best is not None
    return results, best[1]


def _print_table(results: list[ProbeResult]) -> None:
    hdr = f"{'layer':>5} {'pool':>5} {'acc':>6} {'f1':>6} {'prec':>6} {'rec':>6} {'auc':>6} {'fpr':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda x: x.f1, reverse=True):
        print(
            f"{r.layer:>5} {r.pool:>5} {r.accuracy:>6.3f} {r.f1:>6.3f} "
            f"{r.precision:>6.3f} {r.recall:>6.3f} {r.auc:>6.3f} {r.fpr:>6.3f}"
        )


def run(cfg: Config) -> None:
    results, best = train(cfg)
    _print_table(results)
    out = cfg.path("models_dir") / f"probe_{model_slug(cfg.model_config().name)}.pkl"
    best.save(out)
    results_path = cfg.path("models_dir") / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n[trainer] best probe: layer={best.layer} pool={best.pool}")
    print(f"[trainer] saved probe -> {out}")
    print(f"[trainer] saved metrics -> {results_path}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train probes.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
