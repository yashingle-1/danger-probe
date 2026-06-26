# danger-probe

**Detect dangerous intent inside an LLM's activations — before it shows up in the output.**

Most safety filters read the model's *output text*. `danger-probe` reads the model's
*internal residual stream*: a lightweight linear probe trained on hidden activations
flags harmful intent at mid-network layers, in two modes:

- **Mode A — pre-generation:** classify a prompt in one forward pass and decide
  `SAFE` / `WARN` / `BLOCK` before generating a single token.
- **Mode B — streaming:** probe the residual stream during generation and halt
  the moment `P(dangerous)` crosses the block threshold.

It extends **Representation Engineering** (Zou et al., 2023) with a runtime monitor
and — the headline result — a **leave-one-category-out generalization eval**: train
the probe on some attack framings, then test whether it still catches harm under a
framing it never saw, head-to-head against an off-the-shelf guard model.

> Status: research / portfolio project. Defensive eval tooling.

---

## Why this is interesting

- **Earlier than output filters.** The danger signal is present in activations before
  the harmful tokens are produced.
- **Harder to dodge.** Jailbreak and social-engineering *framings* change the surface
  text but not the underlying intent — exactly where keyword filters fail and
  activation probes should hold up.
- **Cheap.** A logistic-regression probe over cached activations; no fine-tuning.

---

## Architecture

```
prompt ── HookedTransformer (TransformerLens) ── residual stream @ layers L
                                                        │
                              pool (last-token / mean)  │
                                                        ▼
                                   StandardScaler → LogisticRegression probe
                                                        │
                                          P(dangerous) ─┴─ SAFE / WARN / BLOCK → audit log
```

Two threat axes, kept as separate probes (different activation directions):
**harm** (primary) and **deception** (secondary, week 2+).

---

## Install

Two environments, because the heavy model needs PyTorch:

**GPU box (full pipeline) — use Python 3.11 or 3.12** (torch/transformer_lens have no
3.13/3.14 wheels yet):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Llama-3.1 is gated:
huggingface-cli login
```

**CPU only (dataset + train-on-cached-activations + eval + viz):** the probe, splits,
baselines, benchmark and plots are pure numpy/sklearn/matplotlib and run anywhere.

---

## Quickstart

```bash
# 1. Build a balanced, deduped dataset (HarmBench/AdvBench + benign Alpaca)
python -m dangerprobe.data.build_dataset

# 2. Extract residual-stream activations  (GPU box; edit config.yaml: active: primary)
python -m dangerprobe.probing.extractor

# 3. Train a probe per (layer, pooling); best is saved by test F1
python -m dangerprobe.probing.trainer

# 4. Benchmark vs baselines
python -m dangerprobe.eval.benchmark --guard toxic_bert

# 5. Visualize the activation space
python -m dangerprobe.viz.activation_space --pool mean

# 6. Live demo (both modes)
python demo/live_demo.py
```

### The generalization experiment (the headline)

In `config.yaml` set:

```yaml
split:
  mode: leave_one_category_out
  holdout_category: manipulation
```

then re-run `trainer` and `benchmark`. The probe is trained on the other framings
and evaluated **only** on the unseen `manipulation` framing — the real test of whether
it learned *intent* rather than *surface keywords*.

---

## Configuration

Everything lives in `config.yaml`. Switch `active: smoke` (GPT-2, CPU) ↔
`active: primary` (Llama-3.1-8B, GPU). Layers, pooling methods, thresholds, dataset
sizes and split mode are all there — no hardcoded model anywhere.

---

## Repo layout

```
dangerprobe/
  config.py            load + validate config.yaml
  data/build_dataset   HarmBench/AdvBench + benign, dedup, balance
  data/splits          random + leave-one-category-out splits
  probing/extractor    TransformerLens hooks (last-token + mean pool)
  probing/trainer      per-layer/pool probe, picks best by F1
  probing/probe        Probe: scaler+clf, save/load, decide()
  probing/monitor      Mode A classify_prompt, Mode B generate_monitored
  baselines/           keyword filter + toxic-bert/Llama Guard
  eval/benchmark       probe vs baselines, in-dist + generalization
  viz/activation_space PCA/t-SNE + per-layer separability
demo/live_demo.py
tests/test_smoke.py    CPU-only end-to-end (synthetic activations)
```

---

## Tests

```bash
PYTHONPATH=. python tests/test_smoke.py
```

Exercises splits → trainer → probe save/load → benchmark → viz on synthetic
activations (no GPU). Asserts the probe recovers an injected signal.

---

## References

- Zou et al. (2023), *Representation Engineering: A Top-Down Approach to AI Transparency*
- Mazeika et al. (2024), *HarmBench*
- Burns et al. (2022), *Discovering Latent Knowledge in Language Models Without Supervision*
- Nanda & Bloom, *TransformerLens*
