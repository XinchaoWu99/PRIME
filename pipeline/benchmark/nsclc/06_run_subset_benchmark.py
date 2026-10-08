"""Robustness experiment, step 4/4 — scIB benchmark + robustness plots.

Consumes the per-subset ``subset_embeddings.h5ad`` files written by 05 and, for
each subset, computes the scIB benchmark (same metric config as script 02,
incl. the isolated-labels score and recomputed aggregate scores) and writes a
per-subset results CSV. Finally it aggregates all subsets into a long-form table
and plots robustness curves (metric vs. number of datasets, mean +/- sd across
replicates, one line per method), with the original full 12-dataset result as a
reference point.

This script is deliberately SEPARATE from the integration (05): it runs in a
clean, GPU-free process so scib-metrics' JAX/XLA CPU backend can create its
thread pool. (Inside the integration process, which has spawned thousands of
threads, XLA hits the ~4096-thread ulimit and dies.)

Run order:  03 -> 04 -> 05 (integration) -> 06 (this).

Usage:
    python 06_run_subset_benchmark.py        # give the job a single CPU (see the note on XLA below)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Force a clean CPU JAX backend BEFORE importing jax / scib_metrics.
# No GPU (PCR's SVD on the n_cells x 2000 ComBat/MNN/BBKNN matrices would
# exceed the memory of a 32 GB GPU), and raise the open-file limit so XLA's CPU client can
# create its thread pool.
# ---------------------------------------------------------------------------
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Cap glibc malloc arenas: with many BLAS/numba threads, glibc opens up to
# 8*nthreads mmap'd arenas which -- together with XLA's JIT mappings and the
# full 12-embedding benchmark state held in RAM -- exhausts the per-process
# mmap-region limit, making LLVM compilation fail with "Cannot allocate memory".
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
# Single-thread the XLA CPU runtime so it doesn't multiply threads/mappings.
# NOTE: XLA still sizes its LLVM-codegen thread pool to the number of *visible*
# CPUs, so the real fix is to give the job 1 CPU (e.g. SLURM --cpus-per-task=1).
# These env vars pin every other library to 1
# thread to match.
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
from sklearn.metrics import silhouette_samples

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # pipeline/ (common.py)
import common as C  # noqa: E402

CFG = C.load_config()

# ---------------------------------------------------------------------------
# Replace kBET's JIT kernel with an equivalent NumPy implementation.
#
# scib's _kbet is jax.jit'd with shape-dependent inputs, so it compiles a fresh
# XLA kernel for every cell-type cluster (one per label, per embedding). Those
# compiled kernels accumulate as executable mmap regions until the process hits
# the kernel limit vm.max_map_count (65530), which surfaces as
#   "LLVM compilation error: Cannot allocate memory"
#   "Failed to materialize symbols"
# at kbet_per_label -- regardless of --mem or --cpus. The NumPy kernel below
# does the identical chi-square computation with ZERO compilation. Verified to
# match the JAX kernel to floating-point precision.
# ---------------------------------------------------------------------------
import scib_metrics.metrics._kbet as _kbet_mod
from scipy.special import gammainc as _gammainc


def _kbet_numpy(neigh_batch_ids, batches, n_batches):
    neigh = np.asarray(neigh_batch_ids)
    batches = np.asarray(batches)
    nb = int(n_batches)
    expected_freq = np.bincount(batches, minlength=nb).astype(np.float64)
    expected_freq = expected_freq / expected_freq.sum()
    dof = nb - 1
    k = neigh.shape[1]
    observed = np.empty((neigh.shape[0], nb), dtype=np.float64)
    for b in range(nb):
        observed[:, b] = (neigh == b).sum(axis=1)
    expected_counts = expected_freq * k
    test_statistics = np.sum((observed - expected_counts) ** 2 / expected_counts, axis=1)
    p_values = 1.0 - _gammainc(dof / 2.0, test_statistics / 2.0)
    return test_statistics, p_values


_kbet_mod._kbet = _kbet_numpy  # kbet() resolves _kbet from the module at call time

# ============================================================================
# User settings (keep in sync with 03 / 04 / 05)
# ============================================================================
ROBUST_ROOT = Path(CFG["benchmark"]["lung"]["dir"]) / "robustness"
MANIFEST_CSV = ROBUST_ROOT / "subset_manifest.csv"

# Reference (full 12-dataset) benchmark, plotted as the right-most point.
FULL_DATASET_DIR = Path(CFG["benchmark"]["lung"]["dir"])
FULL_RESULTS_CSV = FULL_DATASET_DIR / "benchmark_results_with_isolated_labels.csv"
FULL_N_DATASETS = 12

BATCH_KEY = "batch"
LABEL_KEY = "cell_type"

EMBEDDINGS_FILE = "subset_embeddings.h5ad"          # written by 05
RESULTS_FILE = "benchmark_results_with_isolated_labels.csv"

# Parallelism for neighbor computation (numba threads; not the JAX backend).
# Single-core run -> 1.
NEIGHBOR_N_JOBS = 1

# Embeddings to benchmark (same order/names as script 02).
BENCHMARK_KEYS = [
    "ComBat", "Scanorama", "Harmony", "MNN", "PRIME", "BBKNN",
    "scVI", "Unintegrated", "CCA", "FastMNN", "jPCA", "RPCA",
]
# Pre-integration baseline for pcr_comparison. We pass the "Unintegrated" PCA
# embedding explicitly so scib does NOT try to PCA adata.X -- the saved
# embeddings.h5ad carries only a placeholder X. (In 02 scib computed this PCA
# internally from the unintegrated expression; "Unintegrated" is that same PCA.)
PRE_INTEGRATED_KEY = "Unintegrated"

# Line order of the robustness curves; tab20 colours are assigned in this order, so every method keeps its colour.
CURVE_ORDER = ["BBKNN", "CCA", "ComBat", "PRIME", "FastMNN", "Harmony", "MNN", "RPCA", "Scanorama", "Unintegrated",
               "jPCA", "scVI"]

# Benchmark bookkeeping constants (from script 02).
RESULTS_METRIC_TYPE_ROW = "Metric Type"
AGGREGATE_SCORE_VALUE = "Aggregate score"
BIO_CONSERVATION_COL = "Bio conservation"
BATCH_CORRECTION_COL = "Batch correction"
TOTAL_SCORE_COL = "Total"
ISOLATED_LABELS_COL = "Isolated labels"

# ============================================================================
# Benchmark + isolated labels (from script 02)
# ============================================================================

def run_benchmark(adata: sc.AnnData) -> pd.DataFrame:
    missing = [k for k in BENCHMARK_KEYS if k not in adata.obsm]
    if missing:
        raise ValueError("Missing embeddings in adata.obsm: " + ", ".join(missing))
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
        pre_integrated_embedding_obsm_key=PRE_INTEGRATED_KEY,
        n_jobs=NEIGHBOR_N_JOBS,
    )
    bm.benchmark()
    return bm.get_results()


def _get_isolated_labels(labels, batch, iso_threshold):
    if batch is None:
        return np.unique(labels)
    tmp = pd.DataFrame({"label": labels, "batch": batch}).drop_duplicates()
    batch_per_label = tmp.groupby("label")["batch"].count()
    if iso_threshold is None:
        iso_threshold = int(batch_per_label.min())
    return batch_per_label[batch_per_label <= iso_threshold].index.to_numpy()


def isolated_label_score_single(X, labels, batch=None, rescale=True, iso_threshold=None):
    isolated = _get_isolated_labels(labels, batch, iso_threshold)
    if isolated.size == 0:
        return float("nan")
    try:
        sil_all = silhouette_samples(X, labels, metric="euclidean")
    except ValueError:
        return float("nan")
    if rescale:
        sil_all = (sil_all + 1.0) / 2.0
    per_label = [float(np.mean(sil_all[labels == lbl])) for lbl in isolated]
    return float(np.mean(per_label))


def compute_isolated_label_scores(adata, label_key, embedding_keys, batch_key=None,
                                  rescale=True, iso_threshold=None):
    labels = adata.obs[label_key].to_numpy().astype(str)
    batch = adata.obs[batch_key].to_numpy().astype(str) if batch_key else None
    results = {}
    for key in embedding_keys:
        if key not in adata.obsm:
            results[key] = float("nan")
            continue
        X = adata.obsm[key]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        results[key] = isolated_label_score_single(
            X, labels, batch=batch, rescale=rescale, iso_threshold=iso_threshold
        )
    return pd.DataFrame.from_dict(results, orient="index", columns=[ISOLATED_LABELS_COL])


def _merge_isolated_label_scores(results_df, isolated_df):
    merged = results_df.copy()
    merged[ISOLATED_LABELS_COL] = np.nan
    method_rows = merged.index[merged.index != RESULTS_METRIC_TYPE_ROW]
    merged.loc[method_rows, ISOLATED_LABELS_COL] = (
        isolated_df.reindex(method_rows)[ISOLATED_LABELS_COL].to_numpy()
    )
    if RESULTS_METRIC_TYPE_ROW in merged.index:
        merged.loc[RESULTS_METRIC_TYPE_ROW, ISOLATED_LABELS_COL] = BIO_CONSERVATION_COL
        cols_without_new = [c for c in merged.columns if c != ISOLATED_LABELS_COL]
        metric_type_row = merged.loc[RESULTS_METRIC_TYPE_ROW, cols_without_new]
        aggregate_cols = metric_type_row.index[
            metric_type_row == AGGREGATE_SCORE_VALUE
        ].tolist()
        insert_at = (cols_without_new.index(aggregate_cols[0])
                     if aggregate_cols else len(cols_without_new))
        reordered = (cols_without_new[:insert_at] + [ISOLATED_LABELS_COL]
                     + cols_without_new[insert_at:])
        merged = merged.loc[:, reordered]
    return merged


def _recompute_aggregate_scores(results_df):
    updated = results_df.copy()
    metric_type_row = updated.loc[RESULTS_METRIC_TYPE_ROW]
    method_rows = updated.index[updated.index != RESULTS_METRIC_TYPE_ROW]
    bio_metric_cols = metric_type_row.index[
        metric_type_row == BIO_CONSERVATION_COL
    ].tolist()
    bio_metrics = updated.loc[method_rows, bio_metric_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    batch_scores = pd.to_numeric(
        updated.loc[method_rows, BATCH_CORRECTION_COL], errors="coerce"
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


def benchmark_subset(adata: sc.AnnData) -> pd.DataFrame:
    """Full per-subset benchmark, matching script 02's final table."""
    results_df = run_benchmark(adata)
    isolated_df = compute_isolated_label_scores(
        adata, label_key=LABEL_KEY, embedding_keys=BENCHMARK_KEYS,
        batch_key=BATCH_KEY, rescale=True, iso_threshold=None,
    )
    results_df = _merge_isolated_label_scores(results_df, isolated_df)
    results_df = _recompute_aggregate_scores(results_df)
    return results_df


def process_subset(spec: pd.Series) -> None:
    sub_dir = ROBUST_ROOT / spec["subset_id"]
    out_csv = sub_dir / RESULTS_FILE
    if out_csv.exists():
        print(f"[exists] {spec['subset_id']} -> {out_csv} (skip)")
        return
    emb_file = sub_dir / EMBEDDINGS_FILE
    if not emb_file.exists():
        print(f"[warn] {spec['subset_id']}: {emb_file} not found "
              f"(run 05_run_subset_integration.py first); skipping")
        return

    print(f"\n{'=' * 60}\n  Benchmarking {spec['subset_id']} "
          f"(N={spec['n_datasets']}, rep={spec['replicate']})\n{'=' * 60}")
    adata = sc.read_h5ad(emb_file)
    # Downcast embeddings to float32 (scanpy's native dtype): halves the memory
    # of the n x 2000 ComBat/MNN/BBKNN matrices and their per-embedding copies,
    # which is what tips the process into "Cannot allocate memory" at kBET.
    for key in BENCHMARK_KEYS:
        emb = adata.obsm.get(key)
        if emb is not None and getattr(emb, "dtype", None) == np.float64:
            adata.obsm[key] = np.ascontiguousarray(emb, dtype=np.float32)
    results_df = benchmark_subset(adata)
    results_df.to_csv(out_csv, index=True)
    print(f"[done] wrote {out_csv}")


# ============================================================================
# Aggregation + robustness plots
# ============================================================================

def _results_to_long(results_df: pd.DataFrame, n_datasets, replicate) -> pd.DataFrame:
    df = results_df.drop(index=RESULTS_METRIC_TYPE_ROW)
    keep = [TOTAL_SCORE_COL, BIO_CONSERVATION_COL, BATCH_CORRECTION_COL]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    df = df.reset_index().rename(columns={"index": "Method"})
    long = df.melt(id_vars="Method", var_name="metric", value_name="score")
    long["n_datasets"] = n_datasets
    long["replicate"] = replicate
    return long


def aggregate_results(manifest: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, spec in manifest.iterrows():
        csv = ROBUST_ROOT / spec["subset_id"] / RESULTS_FILE
        if not csv.exists():
            print(f"[warn] missing results for {spec['subset_id']}, skipping")
            continue
        rdf = pd.read_csv(csv, index_col=0)
        frames.append(_results_to_long(rdf, spec["n_datasets"], spec["replicate"]))

    # Reference: original full 12-dataset benchmark.
    if FULL_RESULTS_CSV.exists():
        rdf = pd.read_csv(FULL_RESULTS_CSV, index_col=0)
        frames.append(_results_to_long(rdf, FULL_N_DATASETS, replicate=0))

    if not frames:
        raise RuntimeError("No per-subset results found; run the benchmark first.")

    long = pd.concat(frames, ignore_index=True)
    out = ROBUST_ROOT / "robustness_benchmark_long.csv"
    long.to_csv(out, index=False)
    print(f"\nWrote aggregated long table: {out}")
    return long


def plot_robustness(long: pd.DataFrame) -> None:
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
            print(f"Wrote {out}")
        plt.close(fig)


# ============================================================================

def main() -> None:
    manifest = pd.read_csv(MANIFEST_CSV)
    print(f"Loaded manifest with {len(manifest)} subsets from {MANIFEST_CSV}")

    for _, spec in manifest.iterrows():
        process_subset(spec)

    long = aggregate_results(manifest)
    plot_robustness(long)
    print("\nRobustness benchmark complete.")


if __name__ == "__main__":
    main()
