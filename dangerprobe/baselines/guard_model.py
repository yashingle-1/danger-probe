"""Off-the-shelf guard-model baseline.

Default: `unitary/toxic-bert` (light, fast). Optional: Meta Llama Guard (heavier,
gated) for a stronger comparison. Needs transformers + torch, so this runs on the
GPU box, not in the local smoke test.
"""
from __future__ import annotations

import numpy as np


class ToxicBertGuard:
    name = "toxic_bert"

    def __init__(self, device: str = "cpu", batch_size: int = 16):
        from transformers import pipeline

        self.batch_size = batch_size
        self.pipe = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=0 if device == "cuda" else -1,
            top_k=None,
            truncation=True,
        )

    def predict_proba(self, prompts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            for scores in self.pipe(batch):
                # scores: list of {label, score}; take max toxic-ish score
                p = max(s["score"] for s in scores)
                out.append(p)
        return np.array(out, dtype=float)


class LlamaGuard:
    """Meta Llama Guard 3. Gated HF model; stronger but heavier."""

    name = "llama_guard"

    def __init__(self, device: str = "cuda", model_id: str = "meta-llama/Llama-Guard-3-8B"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device
        )

    def _score_one(self, prompt: str) -> float:
        chat = [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(chat, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=10, pad_token_id=0)
        text = self.tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).lower()
        return 1.0 if "unsafe" in text else 0.0

    def predict_proba(self, prompts: list[str]) -> np.ndarray:
        return np.array([self._score_one(p) for p in prompts], dtype=float)


def load_guard(name: str = "toxic_bert", device: str = "cpu"):
    if name == "toxic_bert":
        return ToxicBertGuard(device=device)
    if name == "llama_guard":
        return LlamaGuard(device=device)
    raise ValueError(f"unknown guard: {name}")
