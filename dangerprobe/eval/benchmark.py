"""Head-to-head eval: activation probe vs baselines on the test split.

Honors config.split, so the same harness gives you both:
    split.mode = random                 -> in-distribution numbers
    split.mode = leave_one_category_out -> the generalization headline
        (probe trained on other framings, tested on the held-out one)

Run the trainer with the SAME split mode first, so the probe is trained on the
matching train half.

CLI:
    python -m dangerprobe.eval.benchmark            # probe + keyword
    python -m dangerprobe.eval.benchmark --guard toxic_bert   # + guard model
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dangerprobe.baselines.keyword_filter import KeywordFilter
from dangerprobe.config import Config, load_config
from dangerprobe.data.splits import load_jsonl, make_split
from dangerprobe.probing.extractor import model_slug
from dangerprobe.probing.probe import Probe
from dangerprobe.probing.trainer import _metrics


def _probe_scores(cfg: Config, probe: Probe, test_idx: np.ndarray) -> np.ndarray:
    act_dir = cfg.path("activations_dir") / model_slug(probe.model_name)
    arr = np.load(act_dir / f"layer{probe.layer}_{probe.pool}.npy")
    return probe.proba(arr[test_idx])


def run(cfg: Config, guard: str | None = None) -> dict:
    records = load_jsonl(cfg.dataset_out_path())
    prompts = [r["prompt"] for r in records]
    labels = np.array([r["label"] for r in records])
    train_idx, test_idx = make_split(records, cfg)
    y = labels[test_idx]
    test_prompts = [prompts[i] for i in test_idx]

    print(f"[benchmark] split={cfg.split['mode']}  test n={len(test_idx)} "
          f"(dangerous={int(y.sum())}, benign={int((y == 0).sum())})")

    rows: dict[str, dict] = {}

    # --- activation probe ---
    probe_path = cfg.path("models_dir") / f"probe_{model_slug(cfg.model_config().name)}.pkl"
    if probe_path.exists():
        probe = Probe.load(probe_path)
        rows[f"probe(L{probe.layer},{probe.pool})"] = _metrics(y, _probe_scores(cfg, probe, test_idx))
    else:
        print(f"[benchmark] no probe at {probe_path}; skipping probe row.")

    # --- keyword baseline (always available) ---
    rows["keyword_filter"] = _metrics(y, KeywordFilter().predict_proba(test_prompts))

    # --- optional guard model (needs torch/transformers) ---
    if guard:
        from dangerprobe.baselines.guard_model import load_guard

        dev = cfg.model_config().device
        rows[guard] = _metrics(y, load_guard(guard, device=dev).predict_proba(test_prompts))

    _print_table(rows)
    out = cfg.path("models_dir") / f"benchmark_{cfg.split['mode']}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[benchmark] saved -> {out}")
    return rows


def _print_table(rows: dict[str, dict]) -> None:
    cols = ["accuracy", "f1", "precision", "recall", "auc", "fpr"]
    name_w = max(len(k) for k in rows) + 2
    hdr = f"{'method':<{name_w}}" + "".join(f"{c:>8}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for name, m in sorted(rows.items(), key=lambda kv: kv[1]["f1"], reverse=True):
        print(f"{name:<{name_w}}" + "".join(f"{m[c]:>8.3f}" for c in cols))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Benchmark probe vs baselines.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--guard", default=None, choices=["toxic_bert", "llama_guard"])
    args = ap.parse_args()
    run(load_config(args.config), guard=args.guard)


if __name__ == "__main__":
    main()
