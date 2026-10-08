"""Style and helpers shared by the analysis figures: palette, Arial with editable text in the PDFs, and the save / panel /
load / sweep / tick and bar helpers. Result tables are read from `out_dir` of the pipeline configuration; figures are
written to FIG (set per notebook with set_figure_dir).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import C, load as _load_config          # C = pipeline/common.py
from .display import rel
from .names import rename_methods

CFG = _load_config()
OUT = Path(CFG["out_dir"])
FIG = Path(CFG["figure_dir"])


def set_figure_dir(path) -> None:
    global FIG
    FIG = Path(path)


SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY, INK, INK2, GRID = "#b7b6b1", "#0b0b0b", "#52514e", "#e8e7e4"
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = mpl.colors.LinearSegmentedColormap.from_list("seq", ["#cde2fb"] + BLUE_RAMP)
DIV = mpl.colors.LinearSegmentedColormap.from_list("div", ["#a3302f", "#e34948", "#f0efec", "#3987e5", "#104281"])
# Arial (first available of the list below); TrueType (Type 42) embedding keeps every label an editable text object
# in Illustrator / Acrobat. Fonts of the user font folder are registered explicitly (matplotlib may not scan it).
for _f in sorted((Path.home() / ".local/share/fonts").glob("Arial*.[Tt][Tt][Ff]")):
    mpl.font_manager.fontManager.addfont(str(_f))
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.5, "lines.markersize": 4,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
}


def apply_style() -> None:
    """Default rcParams + the style above."""
    mpl.rcdefaults()
    mpl.rcParams.update(STYLE)


apply_style()


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", rel(FIG / name), flush=True)


def panel(ax, letter):
    ax.text(-0.18, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def load(pattern):
    fs = sorted(glob.glob(str(OUT / pattern)))
    if not fs:
        return None
    return rename_methods(pd.concat([pd.read_csv(f, sep="\t").assign(_file=f) for f in fs], ignore_index=True))


def sweep(df, default, param):
    """Rows where every parameter except `param` equals the default setting."""
    p = df["params"].map(json.loads)
    keep = p.map(lambda d: all(d.get(k) == v for k, v in default.items() if k != param))
    sub = df[keep].copy()
    sub["value"] = p[keep].map(lambda d: d[param])
    return sub


def plain_log_ticks(ax, values, base=10, keep=None):
    """Log axis labelled with tested values at least x2.2 apart; `keep` (the default) is always labelled."""
    ax.set_xscale("log", base=base)
    ticks = [keep] if keep is not None else []
    for v in sorted(values):
        if all(max(v, t) / min(v, t) >= 2.2 for t in ticks):
            ticks.append(v)
    ticks = sorted(ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def line_with_band(ax, sub, metric, color, label):
    g = sub.groupby("value")[metric].agg(["mean", "std"]).sort_index()
    ax.plot(g.index, g["mean"], "-o", color=color, label=label)
    ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.15, lw=0)


def bar_highlight(ax, series, err=None, highlight="PRIME", fmt="{:.2f}", order=None):
    """Horizontal bars (sorted by value unless `order` is given, bottom to top); PRIME in
    slot 1, others grey; every bar labelled with its method and value."""
    s = series.sort_values() if order is None else series.reindex(order)
    colors = [SLOT[0] if highlight in str(i) else GREY for i in s.index]
    ax.barh(range(len(s)), s.values, color=colors, height=0.7,
            xerr=None if err is None else err.reindex(s.index).values, error_kw={"lw": 0.8, "ecolor": INK2})
    ax.set_yticks(range(len(s)), s.index)
    for n, v in enumerate(s.values):
        if np.isfinite(v):
            ax.text(v, n, " " + fmt.format(v), va="center", fontsize=6, color=INK)
    ax.grid(axis="y", visible=False)
