"""NSCLC atlas, lung adenocarcinoma: scIB benchmark of the 12 integration methods, results table, UMAPs (step 3).

Cell-structured script (`#%%` cells). Reads `benchmarked_lung_cancer_integrated.h5ad` (01) and the embeddings of the
Seurat-based methods (`lung_cancer_<method>_corrected.npy`, ../r_methods/run_r_methods.R) from config.yaml
`benchmark.lung.dir`, and writes there:
  benchmarked_lung_cancer_integrated_full.h5ad   all 12 embeddings in obsm
  benchmark_results.csv                           scib-metrics Benchmarker (Leiden NMI / ARI, cLISI, BRAS, iLISI, kBET,
                                                  PCR comparison)
  benchmark_results_with_isolated_labels.csv      + isolated-label silhouette; Bio = mean of the bio metrics,
                                                  Total = 0.6 Bio + 0.4 Batch
  benchmark_results_table.pdf, lung_cancer_<method>_umap_<colour>.png
"""
#%%
from __future__ import annotations

from contextlib import nullcontext
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Colormap
from matplotlib.text import Text
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
from sklearn.metrics import silhouette_samples

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # pipeline/ (common.py)
import common as C  # noqa: E402

CFG = C.load_config()

logger = logging.getLogger(__name__)

# ============================================================================
# Benchmark metrics and plotting constants
# ============================================================================
RESULTS_METRIC_TYPE_ROW = "Metric Type"
AGGREGATE_SCORE_VALUE = "Aggregate score"
BIO_CONSERVATION_COL = "Bio conservation"
BATCH_CORRECTION_COL = "Batch correction"
TOTAL_SCORE_COL = "Total"
ISOLATED_LABELS_COL = "Isolated labels"

# ============================================================================
# User settings (edit these first)
# ============================================================================
#%%
# Dataset folder containing the integrated h5ad with embeddings
DATASET_DIR = Path(CFG["benchmark"]["lung"]["dir"])
dataset_name = "lung_cancer"
INPUT_H5AD = DATASET_DIR / "benchmarked_lung_cancer_integrated.h5ad"

# Metadata keys in adata.obs
BATCH_KEY = "batch"
LABEL_KEY = "cell_type"

# Embeddings to benchmark (must exist in adata.obsm)
BENCHMARK_KEYS = [
    "ComBat",
    "Scanorama",
    "Harmony",
    "MNN",
    "PRIME",
    "BBKNN",
    "scVI",
    "Unintegrated",
    "CCA",
    "FastMNN",
    "jPCA",
    "RPCA",
]

# Parallelism and output files
N_JOBS = 16
RESULTS_CSV = DATASET_DIR / "benchmark_results.csv"
RESULTS_TABLE_PDF = DATASET_DIR / "benchmark_results_table.pdf"

# Plot settings
PLOT_SORT_COL = "Total"
PLOT_FIGSIZE = (11, 6)
PLOT_CELL_CMAP = mpl.cm.PRGn
PLOT_SCORE_CMAP = mpl.cm.OrRd

#%%
# ============================================================================
# Benchmark + output
# ============================================================================

def run_benchmark(
        adata: None
) -> pd.DataFrame:
    missing = [k for k in BENCHMARK_KEYS if k not in adata.obsm]
    if missing:
        raise ValueError(
            "These embedding keys are missing from adata.obsm: "
            + ", ".join(missing)
        )

    bm = Benchmarker(
        adata,
        batch_key=BATCH_KEY,
        label_key=LABEL_KEY,
        bio_conservation_metrics=BioConservation(
            nmi_ari_cluster_labels_leiden=True,
            nmi_ari_cluster_labels_kmeans=False,
            silhouette_label=False,
            isolated_labels=False,
        ),
        batch_correction_metrics=BatchCorrection(graph_connectivity=False),
        embedding_obsm_keys=BENCHMARK_KEYS,
        n_jobs=N_JOBS,
    )

    bm.benchmark()
    results_df = bm.get_results()
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_CSV, index=True)

    # Optional default plot from scib-metrics
    bm.plot_results_table()

    # Publication-style PDF table (quick version)
    # plot_results_table_pdf(results_df, RESULTS_TABLE_PDF)

    return results_df


def plot_results_table_pdf(results_df: pd.DataFrame, out_pdf: Path) -> None:
    # Keep this light and robust: use scib result table values directly.
    metric_type_row_name = "Metric Type"
    aggregate_label = "Aggregate score"

    if metric_type_row_name not in results_df.index:
        raise ValueError(
            "Expected a 'Metric Type' row in benchmark results."
        )

    metric_type_row = results_df.loc[metric_type_row_name]
    score_cols = metric_type_row.index[metric_type_row == aggregate_label].tolist()
    if not score_cols:
        raise ValueError("No aggregate score columns found in results table.")

    plot_df = results_df.drop(index=metric_type_row_name).astype(float)

    if PLOT_SORT_COL in plot_df.columns:
        plot_df = plot_df.sort_values(PLOT_SORT_COL, ascending=False)
    else:
        plot_df = plot_df.sort_values(score_cols[0], ascending=False)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    im = ax.imshow(plot_df.values, aspect="auto", cmap=PLOT_CELL_CMAP)

    ax.set_xticks(range(plot_df.shape[1]))
    ax.set_xticklabels(plot_df.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index, fontsize=9)
    ax.set_title("scIB Benchmark Results")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel("Score", rotation=270, labelpad=15)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "Arial",
    }):
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# ============================================================================
# Isolated labels computation
# ============================================================================

def _get_isolated_labels(
    labels: np.ndarray,
    batch: np.ndarray | None,
    iso_threshold: int | None,
) -> np.ndarray:
    if batch is None:
        return np.unique(labels)

    tmp = pd.DataFrame({"label": labels, "batch": batch}).drop_duplicates()
    batch_per_label = tmp.groupby("label")["batch"].count()

    if iso_threshold is None:
        iso_threshold = int(batch_per_label.min())

    logger.info("Isolated label threshold: <= %s batch(es) per label", iso_threshold)
    isolated = batch_per_label[batch_per_label <= iso_threshold].index.to_numpy()

    if isolated.size == 0:
        logger.warning("No isolated labels found with threshold=%s", iso_threshold)

    return isolated


def isolated_label_score_single(
    X: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray | None = None,
    rescale: bool = True,
    iso_threshold: int | None = None,
) -> float:
    isolated = _get_isolated_labels(labels, batch, iso_threshold)
    if isolated.size == 0:
        logger.warning("No isolated labels found. Returning NaN.")
        return float("nan")

    try:
        sil_all = silhouette_samples(X, labels, metric="euclidean")
    except ValueError as exc:
        logger.warning("Failed to compute isolated label score: %s", exc)
        return float("nan")

    if rescale:
        sil_all = (sil_all + 1.0) / 2.0

    per_label_scores = []
    for label in isolated:
        mask = labels == label
        per_label_scores.append(float(np.mean(sil_all[mask])))

    return float(np.mean(per_label_scores))


def compute_isolated_label_scores(
    adata: sc.AnnData,
    label_key: str,
    embedding_keys: list[str],
    batch_key: str | None = None,
    rescale: bool = True,
    iso_threshold: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    if label_key not in adata.obs.columns:
        raise ValueError(f"label_key={label_key!r} not found in adata.obs")
    if batch_key is not None and batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key={batch_key!r} not found in adata.obs")

    labels = adata.obs[label_key].to_numpy().astype(str)
    batch = adata.obs[batch_key].to_numpy().astype(str) if batch_key else None

    results = {}
    for key in embedding_keys:
        logger.info("Processing embedding %r for isolated label score...", key)
        if key not in adata.obsm:
            logger.warning("Embedding key %r not found in adata.obsm", key)
            results[key] = float("nan")
            continue

        X = adata.obsm[key]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

        if verbose:
            print(f"[{key}] Computing isolated label score...")

        score = isolated_label_score_single(
            X=X,
            labels=labels,
            batch=batch,
            rescale=rescale,
            iso_threshold=iso_threshold,
        )
        results[key] = score

        if verbose:
            print(f"  score = {score:.4f}")

    return pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=[ISOLATED_LABELS_COL],
    )


def _merge_isolated_label_scores(
    results_df: pd.DataFrame,
    isolated_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = results_df.copy()
    merged[ISOLATED_LABELS_COL] = np.nan

    method_rows = merged.index[merged.index != RESULTS_METRIC_TYPE_ROW]
    merged.loc[method_rows, ISOLATED_LABELS_COL] = (
        isolated_df.reindex(method_rows)[ISOLATED_LABELS_COL].to_numpy()
    )

    if RESULTS_METRIC_TYPE_ROW in merged.index:
        merged.loc[RESULTS_METRIC_TYPE_ROW, ISOLATED_LABELS_COL] = BIO_CONSERVATION_COL
        cols_without_new = [col for col in merged.columns if col != ISOLATED_LABELS_COL]
        metric_type_row = merged.loc[RESULTS_METRIC_TYPE_ROW, cols_without_new]
        aggregate_cols = metric_type_row.index[
            metric_type_row == AGGREGATE_SCORE_VALUE
        ].tolist()
        insert_at = (
            cols_without_new.index(aggregate_cols[0])
            if aggregate_cols
            else len(cols_without_new)
        )
        reordered_cols = (
            cols_without_new[:insert_at]
            + [ISOLATED_LABELS_COL]
            + cols_without_new[insert_at:]
        )
        merged = merged.loc[:, reordered_cols]

    return merged


def _recompute_aggregate_scores(results_df: pd.DataFrame) -> pd.DataFrame:
    if RESULTS_METRIC_TYPE_ROW not in results_df.index:
        raise ValueError("results_df must include the Metric Type row.")

    updated = results_df.copy()
    metric_type_row = updated.loc[RESULTS_METRIC_TYPE_ROW]
    method_rows = updated.index[updated.index != RESULTS_METRIC_TYPE_ROW]

    bio_metric_cols = metric_type_row.index[
        metric_type_row == BIO_CONSERVATION_COL
    ].tolist()
    bio_metrics = updated.loc[method_rows, bio_metric_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )
    batch_scores = pd.to_numeric(
        updated.loc[method_rows, BATCH_CORRECTION_COL],
        errors="coerce",
    )

    updated.loc[method_rows, BIO_CONSERVATION_COL] = bio_metrics.mean(axis=1)
    updated.loc[method_rows, TOTAL_SCORE_COL] = (
        0.6 * pd.to_numeric(updated.loc[method_rows, BIO_CONSERVATION_COL], errors="coerce")
        + 0.4 * batch_scores
    )

    updated.loc[RESULTS_METRIC_TYPE_ROW, BATCH_CORRECTION_COL] = AGGREGATE_SCORE_VALUE
    updated.loc[RESULTS_METRIC_TYPE_ROW, BIO_CONSERVATION_COL] = AGGREGATE_SCORE_VALUE
    updated.loc[RESULTS_METRIC_TYPE_ROW, TOTAL_SCORE_COL] = AGGREGATE_SCORE_VALUE
    return updated

#%%
# ============================================================================
# Benchmark table plotting
# ============================================================================

def _validate_results_frame(df: pd.DataFrame) -> None:
    if RESULTS_METRIC_TYPE_ROW not in df.index:
        raise ValueError(
            "Input DataFrame must include a 'Metric Type' row, as returned by "
            "Benchmarker.get_results()."
        )


def _resolve_score_columns(df: pd.DataFrame) -> list[str]:
    metric_type_row = df.loc[RESULTS_METRIC_TYPE_ROW]
    score_cols = metric_type_row.index[metric_type_row == AGGREGATE_SCORE_VALUE].tolist()
    if not score_cols:
        raise ValueError(
            "Input DataFrame must include at least one aggregate score column where "
            "df.loc['Metric Type', col] == 'Aggregate score'."
        )
    return score_cols


def _resolve_sort_col(score_cols: list[str], sort_col: str | None) -> str:
    if sort_col is not None:
        if sort_col not in score_cols:
            raise ValueError(
                f"sort_col {sort_col!r} must be one of the aggregate score columns: "
                f"{score_cols!r}."
            )
        return sort_col

    for candidate in ("Total", "Batch correction", "Bio conservation"):
        if candidate in score_cols:
            return candidate
    return score_cols[0]


def _build_cell_cmap(
    plot_df: pd.DataFrame,
    col: str,
    cell_cmap: Colormap,
    cell_num_stds: float,
):
    return normed_cmap(plot_df[col], cmap=cell_cmap, num_stds=cell_num_stds)


def _default_figsize(df: pd.DataFrame, num_methods: int) -> tuple[float, float]:
    return (len(df.columns) * 1.25, 3 + 0.3 * num_methods)


def _set_figure_text_font_family(fig: Figure, font_family: str) -> None:
    for text_artist in fig.findobj(match=Text):
        text_artist.set_fontfamily(font_family)


def plot_scib_results_table_repro(
    df: pd.DataFrame,
    cell_cmap: Colormap = mpl.cm.PRGn,
    score_cmap: Colormap = mpl.cm.YlGnBu,
    cell_num_stds: float = 2.5,
    sort_col: str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 300,
    figsize: tuple[float, float] | None = None,
    svg_text_as_text: bool = True,
) -> tuple[Figure, Axes, Table]:
    """Reproduce scib-metrics' benchmark results table without using its plotting API.

    Parameters
    ----------
    df
        Results table with the same structure as ``Benchmarker.get_results()``.
        Rows are methods plus one ``"Metric Type"`` row.
    cell_cmap
        Colormap used for the metric cells.
    score_cmap
        Colormap used for aggregate score bars.
    cell_num_stds
        Passed through to ``plottable.cmap.normed_cmap`` for metric-cell scaling.
    sort_col
        Aggregate column used for sorting. Defaults to ``Total``, then
        ``Batch correction``, then ``Bio conservation``; otherwise the first
        aggregate column.
    show
        Whether to call ``plt.show()``.
    save_path
        Optional target path for saving the figure. Any format supported by
        Matplotlib is accepted.
    dpi
        Save DPI when ``save_path`` is provided.
    figsize
        Optional explicit figure size. Defaults to the same shape heuristic used
        by ``scib-metrics``.
    svg_text_as_text
        When ``True``, save SVG text as editable text objects by using
        ``svg.fonttype = 'none'`` during figure creation.
    """

    _validate_results_frame(df)

    results_df = df.copy()
    score_cols = _resolve_score_columns(results_df)
    resolved_sort_col = _resolve_sort_col(score_cols, sort_col)

    plot_df = results_df.drop(index=RESULTS_METRIC_TYPE_ROW)
    plot_df = plot_df.sort_values(by=resolved_sort_col, ascending=False).astype(np.float64)
    plot_df["Method"] = plot_df.index

    metric_type_row = results_df.loc[RESULTS_METRIC_TYPE_ROW]
    other_cols = metric_type_row.index[metric_type_row != AGGREGATE_SCORE_VALUE].tolist()
    num_methods = plot_df.shape[0]

    column_definitions = [
        ColumnDefinition(
            "Method",
            width=1.5,
            textprops={"ha": "left", "weight": "bold"},
        )
    ]
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=1,
            textprops={
                "ha": "center",
                "bbox": {"boxstyle": "circle", "pad": 0.25},
            },
            cmap=_build_cell_cmap(plot_df, col, cell_cmap=cell_cmap, cell_num_stds=cell_num_stds),
            group=metric_type_row[col],
            formatter="{:.2f}",
        )
        for col in other_cols
    ]
    column_definitions += [
        ColumnDefinition(
            col,
            width=1,
            title=col.replace(" ", "\n", 1),
            plot_fn=bar,
            plot_kw={
                "cmap": score_cmap,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
            },
            group=metric_type_row[col],
            border="left" if i == 0 else None,
        )
        for i, col in enumerate(score_cols)
    ]

    rc_context = mpl.rc_context({"svg.fonttype": "none"}) if svg_text_as_text else nullcontext()
    with rc_context:
        fig, ax = plt.subplots(figsize=figsize or _default_figsize(results_df, num_methods))
        table = Table(
            plot_df,
            cell_kw={"linewidth": 0, "edgecolor": "k"},
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, facecolor=ax.get_facecolor(), dpi=dpi)

    if show:
        plt.show()

    return fig, ax, table


def save_scib_results_publication_pdf(
    fig: Figure,
    output_path: str | Path,
    *,
    font_family: str = "Arial",
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.02,
) -> None:
    """Save an existing figure as a publication-ready PDF with editable text."""

    pdf_path = Path(output_path)
    if pdf_path.suffix.lower() != ".pdf":
        pdf_path = pdf_path.with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    _set_figure_text_font_family(fig, font_family)
    with mpl.rc_context(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": font_family,
        }
    ):
        fig.savefig(
            pdf_path,
            format="pdf",
            facecolor=fig.get_facecolor(),
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )

#%%
import scanpy as sc
import numpy as np

adata = sc.read(INPUT_H5AD)
print("Available embeddings in adata.obsm:")
print(list(adata.obsm.keys()))

# loading the r-based tools embeddings
methods = ["CCA", "FastMNN", "jPCA", "RPCA"]
for method in methods:
    key = f"{dataset_name}_{method.lower()}"
    embedding_npy = DATASET_DIR / f"{key}_corrected.npy"
    if embedding_npy.exists():
        print(f"Loading {key} from {embedding_npy}")
        adata.obsm[method] = np.load(embedding_npy)
    else:
        print(f"Warning: {embedding_npy} not found. Skipping {key}.")

#%%
df = run_benchmark(adata=adata)
print("Benchmark finished.")
print(f"Input:   {INPUT_H5AD}")
print(f"CSV:     {RESULTS_CSV}")
print("Top rows:")
print(df.head())
# %%
adata.write_h5ad(DATASET_DIR / "benchmarked_lung_cancer_integrated_full.h5ad")
# %% run isolated_labels benchmark
results_df = pd.read_csv(RESULTS_CSV, index_col=0)

# Compute isolated label scores for all embeddings
isolated_results_df = compute_isolated_label_scores(
    adata,
    label_key=LABEL_KEY,
    embedding_keys=BENCHMARK_KEYS,
    batch_key=BATCH_KEY,
    rescale=True,
    iso_threshold=None,  # Auto-detect threshold
    verbose=True,
)

# Merge isolated label scores into the benchmark results
results_df = _merge_isolated_label_scores(results_df, isolated_results_df)

# Recompute aggregate scores (Total, Bio conservation, Batch correction)
results_df = _recompute_aggregate_scores(results_df)

print(results_df)

# %% save the updated results table with isolated labels and new aggregate scores
updated_csv = DATASET_DIR / "benchmark_results_with_isolated_labels.csv"
updated_csv.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(updated_csv, index=True)
# %%
updated_csv = DATASET_DIR / "benchmark_results_with_isolated_labels.csv"
saved_benchmark_df = pd.read_csv(updated_csv, index_col=0)

# put the isolated labels column in the third position
results_df_tmp = saved_benchmark_df.copy()
col = results_df_tmp.pop(ISOLATED_LABELS_COL)
results_df_tmp.insert(3, ISOLATED_LABELS_COL, col)
results_df_tmp
saved_benchmark_df = results_df_tmp

# plot the updated results table with isolated labels included
import matplotlib as mpl

table_fig_file = f"{DATASET_DIR}/benchmark_results_table.pdf"
with mpl.rc_context({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Arial",   # Editable text in the exported PDF.
    "figure.dpi": 300,
}):
    fig, ax, table = plot_scib_results_table_repro(
        saved_benchmark_df,
        cell_cmap=mpl.cm.PRGn,
        score_cmap=mpl.cm.OrRd,
        sort_col="Total",
        figsize=(14, 8),
        show=False,
        save_path=None,
    )

    plt.show()

    fig.savefig(
        table_fig_file,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )

# %% load the integrated adata
adata = sc.read_h5ad(f"{DATASET_DIR}/benchmarked_lung_cancer_integrated_full.h5ad")
adata
# %% Draw the umap colored by cell_type and batch
method_list = ["PRIME", "BBKNN", "scVI", "Unintegrated", 
               "CCA", "FastMNN", "jPCA", "RPCA", "MNN", 
               "Harmony", "Scanorama", "ComBat"]

for method in method_list:
    if method in adata.obsm:
        with mpl.rc_context({
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "Arial",   # Editable text in the exported PDF.
            "figure.dpi": 300,
        }):
            sc.pp.neighbors(adata, use_rep=method)
            sc.tl.umap(adata)
            fig, axes = plt.subplots(1, 1, figsize=(5, 5))
            sc.pl.umap(adata, color=[LABEL_KEY], ax=axes,
                       show=False, legend_loc=None, frameon=False,
                       title=f"{method} - Cell Type")
            fig.savefig(f"{DATASET_DIR}/{dataset_name}_{method}_umap_cell_type.png", format="png", bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(1, 1, figsize=(5, 5))
            sc.pl.umap(adata, color=[BATCH_KEY], ax=axes, frameon=False, 
                       show=False, legend_loc=None,
                       title=f"{method} - Batch")
            fig.savefig(f"{DATASET_DIR}/{dataset_name}_{method}_umap_batch.png", format="png", bbox_inches="tight")
            plt.close(fig)
    else:
        print(f"Warning: {method} embedding not found in adata.obsm. Skipping UMAP for this method.")
        
