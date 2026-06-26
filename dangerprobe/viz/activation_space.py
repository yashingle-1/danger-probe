"""Visualize the activation space: do safe/dangerous prompts separate, and where?

Produces two figures in config.paths.figures_dir:
    pca_<pool>.png        PCA scatter per layer, colored by label
    separation_<pool>.png cross-validated AUC per layer (which layer separates best)

t-SNE is optional (--tsne); it's slow, so PCA is the default.

CLI:
    python -m dangerprobe.viz.activation_space
    python -m dangerprobe.viz.activation_space --pool last --tsne
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dangerprobe.config import Config, load_config
from dangerprobe.probing.extractor import model_slug


def _load(act_dir: Path, layer: int, pool: str) -> np.ndarray:
    return np.load(act_dir / f"layer{layer}_{pool}.npy")


def _cv_auc(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    folds = max(2, min(5, n_pos, n_neg))
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    scores = cross_val_score(pipe, X, y, cv=folds, scoring="roc_auc")
    return float(scores.mean())


def plot_pca(cfg: Config, pool: str, use_tsne: bool = False) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    mc = cfg.model_config()
    act_dir = cfg.path("activations_dir") / model_slug(mc.name)
    labels = np.load(act_dir / "labels.npy")
    layers = mc.layers
    fig_dir = cfg.path("figures_dir")
    fig_dir.mkdir(parents=True, exist_ok=True)

    ncols = len(layers)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)
    for ax, L in zip(axes[0], layers):
        X = StandardScaler().fit_transform(_load(act_dir, L, pool))
        if use_tsne:
            from sklearn.manifold import TSNE

            perp = max(5, min(30, (len(X) - 1) // 3))
            XY = TSNE(n_components=2, perplexity=perp, init="pca",
                      random_state=cfg.seed).fit_transform(X)
            title = f"layer {L} (t-SNE)"
        else:
            XY = PCA(n_components=2, random_state=cfg.seed).fit_transform(X)
            title = f"layer {L} (PCA)"
        for lbl, color, name in [(0, "tab:blue", "benign"), (1, "tab:red", "dangerous")]:
            m = labels == lbl
            ax.scatter(XY[m, 0], XY[m, 1], s=12, alpha=0.6, c=color, label=name)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0][0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{mc.name} — activation space ({pool}-pool)")
    fig.tight_layout()
    method = "tsne" if use_tsne else "pca"
    out = fig_dir / f"{method}_{pool}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[viz] wrote {out}")


def plot_separation(cfg: Config, pool: str) -> None:
    import matplotlib.pyplot as plt

    mc = cfg.model_config()
    act_dir = cfg.path("activations_dir") / model_slug(mc.name)
    labels = np.load(act_dir / "labels.npy")
    layers = mc.layers
    aucs = [_cv_auc(_load(act_dir, L, pool), labels, cfg.seed) for L in layers]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar([str(L) for L in layers], aucs, color="tab:purple")
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("layer")
    ax.set_ylabel("CV AUC")
    ax.set_title(f"Separability by layer ({pool}-pool)")
    for i, a in enumerate(aucs):
        ax.text(i, a + 0.005, f"{a:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = cfg.path("figures_dir") / f"separation_{pool}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[viz] wrote {out}  (AUC per layer: {dict(zip(layers, [round(a,3) for a in aucs]))})")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Visualize activation space.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--pool", default="mean", choices=["last", "mean"])
    ap.add_argument("--tsne", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    plot_separation(cfg, args.pool)
    plot_pca(cfg, args.pool, use_tsne=args.tsne)


if __name__ == "__main__":
    main()
