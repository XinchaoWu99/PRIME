"""NSCLC (lung adenocarcinoma) robustness to the number of integrated datasets: per-subset scIB results written by
pipeline/benchmark/nsclc/06_run_subset_benchmark.py (random subsets of 4, 6, 8 and 10 of the 12 datasets, 5 replicates
each).

plot_metric_bars / plot_score_bars: one bar per method (mean +/- SD over the replicates that finished);
plot_robustness_curves: score against the number of datasets, as in 06_run_subset_benchmark.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .names import rename_methods

METHOD_ORDER = ["PRIME", "ComBat", "Scanorama", "Harmony", "MNN", "BBKNN", "scVI", "Unintegrated",
                "CCA", "FastMNN", "jPCA", "RPCA"]
DISPLAY: dict = {}          # display names that differ from the method key
METHOD_COLORS = {"PRIME": "#2a78d6", "ComBat": "#1baf7a", "Scanorama": "#eda100", "Harmony": "#008300",
                 "MNN": "#4a3aa7", "BBKNN": "#e34948", "scVI": "#e87ba4", "Unintegrated": "#eb6834",
                 "CCA": "#8c564b", "FastMNN": "#17becf", "jPCA": "#bcbd22", "RPCA": "#6b7280"}
SCORE_COLORS = {"Batch correction": "#2a78d6", "Bio conservation": "#eb6834", "Total": "#4a3aa7"}
# Line order of the robustness curves; tab20 colours are assigned in this order, so every method keeps its colour.
CURVE_ORDER = ["BBKNN", "CCA", "ComBat", "PRIME", "FastMNN", "Harmony", "MNN", "RPCA", "Scanorama", "Unintegrated",
               "jPCA", "scVI"]
AXIS_GREY, GRID_GREY, INK = "#b7b6b1", "#e0dfd9", "#0b0b0b"


def load_subset_results(robust_root: Path, results_file: str = "benchmark_results_with_isolated_labels.csv") -> pd.DataFrame:
    """Long table (Method, metric, score, n_datasets, replicate) over all subsets with a results table."""
    manifest = pd.read_csv(Path(robust_root) / "subset_manifest.csv")
    frames = []
    for _, spec in manifest.iterrows():
        f = Path(robust_root) / spec["subset_id"] / results_file
        if not f.exists():
            print(f"[info] no results for {spec['subset_id']} (not finished); skipped")
            continue
        df = rename_methods(pd.read_csv(f, index_col=0))
        df = df[df.index.isin(METHOD_ORDER)].apply(pd.to_numeric, errors="coerce")
        long = df.reset_index(names="Method").melt(id_vars="Method", var_name="metric", value_name="score")
        frames.append(long.assign(n_datasets=int(spec["n_datasets"]), replicate=int(spec["replicate"])))
    return pd.concat(frames, ignore_index=True)


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_GREY)
    ax.tick_params(colors=INK)
    ax.set_axisbelow(True)


def plot_metric_bars(long: pd.DataFrame, n: int, metric: str, out_png: Path) -> None:
    """One metric at n datasets: one bar per method (mean +/- SD over replicates)."""
    g = long[(long.n_datasets == n) & (long.metric == metric)].groupby("Method")["score"].agg(["mean", "std"])
    g = g.reindex([m for m in METHOD_ORDER if m in g.index])
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    x = np.arange(len(g))
    ax.bar(x, g["mean"], width=0.68, color=[METHOD_COLORS[m] for m in g.index],
           yerr=g["std"].fillna(0), capsize=4, error_kw={"lw": 1.5, "ecolor": INK, "capthick": 1.5})
    ax.set_xticks(x, [DISPLAY.get(m, m) for m in g.index], rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color=GRID_GREY, lw=0.8)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_score_bars(long: pd.DataFrame, n: int, out_png: Path) -> None:
    """Batch correction, bio conservation and total score at n datasets, grouped by method."""
    scores = list(SCORE_COLORS)
    sub = long[(long.n_datasets == n) & long.metric.isin(scores)]
    g = sub.groupby(["Method", "metric"])["score"].agg(["mean", "std"]).unstack("metric")
    g = g.reindex([m for m in METHOD_ORDER if m in g.index])
    fig, ax = plt.subplots(figsize=(13.8, 6.2))
    x = np.arange(len(g))
    w = 0.24
    for j, s in enumerate(scores):
        ax.bar(x + (j - 1) * w, g[("mean", s)], width=w, color=SCORE_COLORS[s], label=s,
               yerr=g[("std", s)].fillna(0), capsize=3, error_kw={"lw": 1.5, "ecolor": INK, "capthick": 1.5})
    lo = np.nanmin((g["mean"] - g["std"].fillna(0)).values)
    ax.set_ylim(max(0.0, lo - 0.05), None)
    ax.set_xticks(x, [DISPLAY.get(m, m) for m in g.index], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- copied from 008_run_subset_benchmark.py
TOTAL_SCORE_COL, BIO_CONSERVATION_COL, BATCH_CORRECTION_COL = "Total", "Bio conservation", "Batch correction"


def plot_robustness_curves(long: pd.DataFrame, out_dir: Path) -> None:
    """Score vs number of integrated datasets (mean +/- SD over replicates); written as robustness_curves.pdf/png."""
    ROBUST_ROOT = Path(out_dir)
    metrics = [TOTAL_SCORE_COL, BIO_CONSERVATION_COL, BATCH_CORRECTION_COL]
    metrics = [m for m in metrics if m in long["metric"].unique()]
    present = set(long["Method"].unique())
    methods = [m for m in CURVE_ORDER if m in present] + sorted(present - set(CURVE_ORDER))
    cmap = plt.get_cmap("tab20")
    colors = {m: cmap(i % 20) for i, m in enumerate(methods)}

    with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "Arial", "figure.dpi": 300}):
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5),
                                 squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            sub = long[long["metric"] == metric]
            grouped = sub.groupby(["Method", "n_datasets"])["score"]
            stats = grouped.agg(["mean", "std"]).reset_index()
            for method in methods:
                ms = stats[stats["Method"] == method].sort_values("n_datasets")
                if ms.empty:
                    continue
                ax.errorbar(
                    ms["n_datasets"], ms["mean"], yerr=ms["std"].fillna(0.0),
                    marker="o", capsize=3, label=method, color=colors[method],
                    linewidth=1.5, markersize=4,
                )
            ax.set_title(metric)
            ax.set_xlabel("Number of datasets integrated")
            ax.set_ylabel(f"{metric} score")
            ax.set_xticks(sorted(long["n_datasets"].unique()))
            ax.grid(True, alpha=0.3)
        axes[0][-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                           fontsize=8, frameon=False)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            out = ROBUST_ROOT / f"robustness_curves.{ext}"
            fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
            print(f"Wrote {out.name}")
        plt.close(fig)
