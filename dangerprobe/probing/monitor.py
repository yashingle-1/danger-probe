"""Runtime monitor — two modes over a loaded probe.

Mode A  classify_prompt(prompt)      : one forward pass, decide before generating.
Mode B  generate_monitored(prompt)   : greedy generation that halts the moment
                                       P(dangerous) at the generating position
                                       crosses the block threshold.

Both write an audit record to config.paths.audit_dir.

Mode B reads the residual at the *current* position, so it pairs naturally with
a probe trained on pool="last". If the loaded probe used pool="mean" a caveat is
printed (still runs, but last-token scoring is the principled signal here).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from dangerprobe.config import Config, load_config
from dangerprobe.probing.extractor import ActivationExtractor, model_slug
from dangerprobe.probing.probe import Probe


class Monitor:
    def __init__(self, cfg: Config, probe: Probe, extractor: ActivationExtractor):
        self.cfg = cfg
        self.probe = probe
        self.extractor = extractor
        self.audit_dir = cfg.path("audit_dir")
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.hook = f"blocks.{probe.layer}.hook_resid_post"

    # --- audit -----------------------------------------------------------
    def _audit(self, payload: dict) -> Path:
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
        path = self.audit_dir / f"{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    # --- Mode A ----------------------------------------------------------
    def classify_prompt(self, prompt: str) -> dict:
        single = self.extractor.extract_single(prompt)
        vec = single[(self.probe.layer, self.probe.pool)]
        p = self.probe.proba_one(vec)
        decision = self.probe.decide(p)
        result = {
            "mode": "A_prompt",
            "prompt": prompt,
            "p_dangerous": round(p, 4),
            "decision": decision,
            "layer": self.probe.layer,
            "pool": self.probe.pool,
        }
        result["audit_log"] = str(self._audit(result))
        print(f"[DANGER-PROBE] {self._badge(decision)} P(dangerous)={p:.3f} at layer {self.probe.layer}")
        return result

    # --- Mode B ----------------------------------------------------------
    def generate_monitored(self, prompt: str, max_new_tokens: int = 64) -> dict:
        import torch

        if self.probe.pool != "last":
            print(
                f"[monitor] note: probe pool='{self.probe.pool}'; Mode B scores the "
                f"last position. A pool='last' probe is recommended for streaming."
            )
        model = self.extractor.model
        text = self.extractor._format(prompt)
        tokens = model.to_tokens(text)
        eos = model.tokenizer.eos_token_id
        max_p = 0.0
        halted = False
        generated = []

        for step in range(max_new_tokens):
            with torch.no_grad():
                logits, cache = model.run_with_cache(
                    tokens, names_filter=lambda n: n == self.hook
                )
            resid_last = cache[self.hook][0, -1].float().cpu().numpy()
            p = self.probe.proba_one(resid_last)
            max_p = max(max_p, p)
            if self.probe.decide(p) == "BLOCK":
                halted = True
                break
            next_tok = int(logits[0, -1].argmax())
            generated.append(next_tok)
            if next_tok == eos:
                break
            tokens = torch.cat(
                [tokens, torch.tensor([[next_tok]], device=tokens.device)], dim=1
            )

        text_out = model.tokenizer.decode(generated) if generated else ""
        decision = "BLOCK" if halted else "SAFE"
        result = {
            "mode": "B_streaming",
            "prompt": prompt,
            "max_p_dangerous": round(max_p, 4),
            "decision": decision,
            "halted_at_token": len(generated) if halted else None,
            "output": text_out,
            "layer": self.probe.layer,
        }
        result["audit_log"] = str(self._audit(result))
        print(
            f"[DANGER-PROBE] {self._badge(decision)} max P(dangerous)={max_p:.3f} "
            f"{'(halted mid-generation)' if halted else ''}"
        )
        return result

    @staticmethod
    def _badge(decision: str) -> str:
        return {"SAFE": "[SAFE]", "WARN": "[WARN]", "BLOCK": "[BLOCK]"}[decision]


def load_monitor(cfg: Config) -> Monitor:
    mc = cfg.model_config()
    probe_path = cfg.path("models_dir") / f"probe_{model_slug(mc.name)}.pkl"
    probe = Probe.load(probe_path)
    extractor = ActivationExtractor(mc)
    return Monitor(cfg, probe, extractor)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run the monitor on a prompt.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--mode", choices=["A", "B"], default="A")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mon = load_monitor(cfg)
    if args.mode == "A":
        mon.classify_prompt(args.prompt)
    else:
        mon.generate_monitored(args.prompt)


if __name__ == "__main__":
    main()
