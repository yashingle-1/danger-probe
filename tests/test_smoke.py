"""Smoke test for the CPU-runnable pipeline (no torch / no GPU).

The activation extractor needs torch+transformer_lens (GPU box). Everything
downstream — splits, trainer, probe save/load, benchmark, baselines, viz — is
pure numpy/sklearn/matplotlib and is exercised here against *synthetic*
activations laid out exactly like the real extractor's output.

A learnable signal is injected into the dangerous class so a working trainer
must score high F1; that proves the train/eval wiring, not model quality.

Run:
    PYTHONPATH=. python tests/test_smoke.py
"""
from __future__ import annotations

import json

import numpy as np

from dangerprobe.config import load_config
from dangerprobe.data.splits import load_jsonl
from dangerprobe.probing.extractor import model_slug


def _fabricate_activations(cfg, d: int = 64) -> None:
    """Write synthetic layer{L}_{pool}.npy + labels.npy + meta.json."""
    mc = cfg.model_config()
    records = load_jsonl(cfg.dataset_out_path())
    labels = np.array([r["label"] for r in records])
    n = len(records)
    rng = np.random.default_rng(cfg.seed)

    act_dir = cfg.path("activations_dir") / model_slug(mc.name)
    act_dir.mkdir(parents=True, exist_ok=True)

    # Signal grows with layer depth -> deeper layers separate better (realistic).
    for li, L in enumerate(mc.layers):
        strength = 0.6 + 0.3 * li
        for pool in cfg.pooling_methods:
            X = rng.standard_normal((n, d)).astype(np.float32)
            X[labels == 1, :8] += strength  # dangerous shifted on first 8 dims
            np.save(act_dir / f"layer{L}_{pool}.npy", X)

    np.save(act_dir / "labels.npy", labels)
    meta = {
        "model": mc.name,
        "layers": mc.layers,
        "pools": cfg.pooling_methods,
        "n": int(n),
        "categories": [r["threat_category"] for r in records],
    }
    with open(act_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[smoke] fabricated activations for {n} records at {act_dir}")


def main() -> None:
    from dangerprobe.eval import benchmark
    from dangerprobe.probing import trainer
    from dangerprobe.viz import activation_space

    cfg = load_config()

    print("\n=== fabricate activations ===")
    _fabricate_activations(cfg)

    print("\n=== train probes ===")
    results, best = trainer.train(cfg)
    trainer._print_table(results)
    out = cfg.path("models_dir") / f"probe_{model_slug(cfg.model_config().name)}.pkl"
    best.save(out)
    print(f"[smoke] saved probe -> {out}")

    print("\n=== benchmark (probe vs keyword) ===")
    rows = benchmark.run(cfg)

    print("\n=== viz ===")
    activation_space.plot_separation(cfg, pool="mean")
    activation_space.plot_pca(cfg, pool="mean")

    # assertions: trainer should learn the injected signal
    best_f1 = max(r.f1 for r in results)
    assert best_f1 > 0.8, f"probe F1 too low ({best_f1:.3f}) — train path broken"
    assert "keyword_filter" in rows, "keyword baseline missing from benchmark"
    print(f"\n[smoke] PASS — best probe F1={best_f1:.3f}")


if __name__ == "__main__":
    main()
