"""Extract residual-stream activations with TransformerLens.

Runs on the GPU box (Llama-3.1-8B) or locally as a GPT-2 smoke test. Saves one
array per (layer, pooling) so the trainer can sweep them without re-running the
model.

Saved layout (under config.paths.activations_dir / <model_slug>/):
    layer{L}_{pool}.npy   float array (N, d_model)
    labels.npy            int array   (N,)
    meta.json             {model, layers, pools, categories, n}

CLI:
    python -m dangerprobe.probing.extractor
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dangerprobe.config import Config, ModelConfig, load_config
from dangerprobe.data.splits import load_jsonl

_DTYPES = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}


def model_slug(name: str) -> str:
    return name.replace("/", "_")


def _resolve_dtype(name: str):
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


class ActivationExtractor:
    """Loads a HookedTransformer and pulls pooled residual-stream vectors."""

    def __init__(self, mc: ModelConfig):
        from transformer_lens import HookedTransformer

        self.mc = mc
        self.model = HookedTransformer.from_pretrained(
            mc.name,
            device=mc.device,
            dtype=_resolve_dtype(mc.dtype),
        )
        self.model.eval()
        self.hook_names = {L: f"blocks.{L}.hook_resid_post" for L in mc.layers}

    def _format(self, prompt: str) -> str:
        if self.mc.use_chat_template and self.model.tokenizer.chat_template:
            return self.model.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def extract_single(self, prompt: str) -> dict[tuple[int, str], np.ndarray]:
        """Return {(layer, pool): vector} for one prompt. Used by the monitor too."""
        import torch

        text = self._format(prompt)
        tokens = self.model.to_tokens(text)  # (1, seq)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens, names_filter=lambda n: n in self.hook_names.values()
            )
        out: dict[tuple[int, str], np.ndarray] = {}
        for L, hook in self.hook_names.items():
            resid = cache[hook][0]  # (seq, d_model)
            out[(L, "last")] = resid[-1].float().cpu().numpy()
            out[(L, "mean")] = resid.mean(dim=0).float().cpu().numpy()
        return out

    def extract_dataset(self, records: list[dict], pools: list[str]):
        """Extract all records. Returns (features, labels) where features is
        {(layer, pool): array (N, d)}."""
        feats: dict[tuple[int, str], list[np.ndarray]] = {
            (L, p): [] for L in self.mc.layers for p in pools
        }
        labels: list[int] = []
        n = len(records)
        for i, r in enumerate(records):
            single = self.extract_single(r["prompt"])
            for L in self.mc.layers:
                for p in pools:
                    feats[(L, p)].append(single[(L, p)])
            labels.append(int(r["label"]))
            if (i + 1) % 50 == 0 or i + 1 == n:
                print(f"[extractor] {i + 1}/{n}")
        features = {k: np.vstack(v) for k, v in feats.items()}
        return features, np.array(labels, dtype=np.int64)


def save_activations(
    out_dir: Path,
    features: dict[tuple[int, str], np.ndarray],
    labels: np.ndarray,
    records: list[dict],
    mc: ModelConfig,
    pools: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for (L, p), arr in features.items():
        np.save(out_dir / f"layer{L}_{p}.npy", arr)
    np.save(out_dir / "labels.npy", labels)
    meta = {
        "model": mc.name,
        "layers": mc.layers,
        "pools": pools,
        "n": int(len(labels)),
        "categories": [r["threat_category"] for r in records],
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def run(cfg: Config) -> None:
    mc = cfg.model_config()
    records = load_jsonl(cfg.dataset_out_path())
    print(f"[extractor] model={mc.name} device={mc.device} n={len(records)}")
    extractor = ActivationExtractor(mc)
    features, labels = extractor.extract_dataset(records, cfg.pooling_methods)
    out_dir = cfg.path("activations_dir") / model_slug(mc.name)
    save_activations(out_dir, features, labels, records, mc, cfg.pooling_methods)
    print(f"[extractor] saved activations to {out_dir}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Extract activations.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
