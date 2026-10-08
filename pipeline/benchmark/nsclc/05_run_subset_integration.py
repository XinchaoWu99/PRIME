"""Robustness experiment, step 3/4 — Python integration per subset.

For every subset produced by 03 (and after the R embeddings from 04 are in
place) this script re-runs the script-01 Python integration pipeline from
scratch (normalize -> HVG(2000, batch-aware) -> PCA -> ComBat / Scanorama /
Harmony / MNN / BBKNN / scVI / PRIME), attaches the R-based embeddings
(CCA / FastMNN / jPCA / RPCA from 04), and saves all 12 embeddings to a small
``subset_embeddings.h5ad`` (obsm + batch/label obs only -- no gene matrix).

The metric computation lives in a SEPARATE script (06). They are split on
purpose: the integration step (scVI/torch, scanorama, MNN) spawns thousands of
threads, and if the scIB benchmark (which uses JAX/XLA) runs in that same
bloated process it cannot create its CPU thread pool (hits the ~4096 thread
ulimit) and dies with "Cannot allocate memory" / "Failed to materialize
symbols". Running the benchmark in a clean process (06) avoids this entirely.

Run order:  03 -> 04 -> 05 (this, GPU) -> 06 (benchmark, CPU).

Usage:
    python 05_run_subset_integration.py
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # pipeline/ (common.py)
import common as C  # noqa: E402

CFG = C.load_config()

# ============================================================================
# User settings (keep in sync with 03 / 04 / 06)
# ============================================================================
ROBUST_ROOT = Path(CFG["benchmark"]["lung"]["dir"]) / "robustness"
MANIFEST_CSV = ROBUST_ROOT / "subset_manifest.csv"

BATCH_KEY = "batch"
LABEL_KEY = "cell_type"
R_OUTPUT_PREFIX = "subset"  # produced by 04: subset_<rmethod>_corrected.npy

N_JOBS = 16  # parallelism for the integration step (e.g. MNN)

# Embeddings to save (same order/names as script 02).
BENCHMARK_KEYS = [
    "ComBat", "Scanorama", "Harmony", "MNN", "PRIME", "BBKNN",
    "scVI", "Unintegrated", "CCA", "FastMNN", "jPCA", "RPCA",
]
# R-based methods: obsm key -> 04 file suffix.
R_METHOD_FILES = {"CCA": "cca", "FastMNN": "fastmnn", "jPCA": "jpca", "RPCA": "rpca"}

EMBEDDINGS_FILE = "subset_embeddings.h5ad"  # <- consumed by 06
RESULTS_FILE = "benchmark_results_with_isolated_labels.csv"  # written by 06

# ============================================================================
# Python integration pipeline (mirrors script 01)
# ============================================================================

def integrate_python(adata: ad.AnnData) -> ad.AnnData:
    """Run all Python-based integration methods on a raw-counts subset.

    Returns an AnnData restricted to the 2000 HVGs, with one obsm per method.
    Faithfully reproduces 01_integrate_lung_adenocarcinoma.py.
    """
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata, n_top_genes=2000, flavor="cell_ranger", batch_key="batch"
    )
    sc.tl.pca(adata, n_comps=30, use_highly_variable=True)
    adata = adata[:, adata.var.highly_variable].copy()
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]

    batch_cats = adata.obs.batch.cat.categories

    # --- Scanorama ----------------------------------------------------------
    import scanorama
    adata_list = [adata[adata.obs.batch == b].copy() for b in batch_cats]
    scanorama.integrate_scanpy(adata_list, dimred=100, hvg=2000)
    adata.obsm["Scanorama"] = np.zeros(
        (adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1])
    )
    for i, b in enumerate(batch_cats):
        adata.obsm["Scanorama"][adata.obs.batch == b] = adata_list[i].obsm["X_scanorama"]

    # --- Harmony ------------------------------------------------------------
    from harmonypy import run_harmony
    harmony_out = run_harmony(adata.obsm["X_pca"], adata.obs, "batch")
    harmony_emb = np.asarray(harmony_out.Z_corr)
    if harmony_emb.shape[0] != adata.n_obs:  # harmonypy returns (d, N)
        harmony_emb = harmony_emb.T
    adata.obsm["Harmony"] = harmony_emb

    # --- MNN ----------------------------------------------------------------
    adata_list = [adata[adata.obs.batch == b].copy() for b in batch_cats]
    hvgs = adata.var[adata.var.highly_variable].index.tolist()
    corrected = sc.external.pp.mnn_correct(
        adata_list, var_subset=hvgs, batch_key="batch",
        k=20, sigma=0.1, svd_dim=100, save_raw=True, do_concatenate=True, n_jobs=N_JOBS,
    )
    adata_corrected = ad.concat(corrected[0][0])
    adata_corrected = adata_corrected[adata.obs_names, :]
    adata.obsm["MNN"] = adata_corrected.X.toarray()

    # --- ComBat -------------------------------------------------------------
    adata_combat_expression = sc.pp.combat(adata, key="batch", inplace=False)
    adata.obsm["ComBat"] = np.asarray(adata_combat_expression)

    # --- BBKNN --------------------------------------------------------------
    import bbknn
    adata_highly_var = adata[:, adata.var.highly_variable].copy()
    adata_bbknn = adata_highly_var.copy()
    adata_highly_var.layers["logcounts"] = adata_highly_var.X.copy()
    adata_bbknn.X = adata_highly_var.layers["logcounts"].copy()
    sc.pp.pca(adata_bbknn)
    bbknn.bbknn(adata_bbknn, batch_key="batch", neighbors_within_batch=3)
    adata.obsm["BBKNN"] = adata_bbknn.X.toarray().copy()

    # --- scVI ---------------------------------------------------------------
    import scvi
    adata_scvi = adata_highly_var.copy()
    scvi.model.SCVI.setup_anndata(adata_scvi, layer="count", batch_key="batch")
    model_scvi = scvi.model.SCVI(adata_scvi)
    max_epochs_scvi = int(np.min([round((20000 / adata_scvi.n_obs) * 400), 400]))
    model_scvi.train(max_epochs=max_epochs_scvi)
    adata_scvi.obsm["X_scVI"] = model_scvi.get_latent_representation()
    adata.obsm["scVI"] = adata_scvi.obsm["X_scVI"].copy()

    # --- PRIME --------------------------------------------------------------
    from prime import ensemble_mnn_correct
    adata.obsm["PRIME"] = ensemble_mnn_correct(
        adata_highly_var, batch_key="batch", n_projections=4, target_dim=128,
        k_neighbors=15, consensus_threshold=0.4, sigma=0.1, random_state=42,
        key_added=None, inplace=False,
    )
    return adata


def load_r_embeddings(adata: ad.AnnData, sub_dir: Path) -> None:
    """Attach the R-based embeddings (04 outputs) into adata.obsm in place."""
    for obsm_key, suffix in R_METHOD_FILES.items():
        npy = sub_dir / f"{R_OUTPUT_PREFIX}_{suffix}_corrected.npy"
        if not npy.exists():
            raise FileNotFoundError(
                f"Missing R embedding {npy}. Run 04_run_R_methods_subsets.R first."
            )
        emb = np.load(npy)
        if emb.shape[0] != adata.n_obs:
            raise ValueError(
                f"{npy.name}: {emb.shape[0]} rows != {adata.n_obs} cells in subset"
            )
        adata.obsm[obsm_key] = emb


def save_embeddings(adata: ad.AnnData, emb_file: Path) -> None:
    """Persist just the obsm embeddings + batch/label obs for the 06 benchmark.

    Deliberately lightweight (no gene matrix, no layers) so the benchmark
    process stays small and the CPU XLA JIT has room.
    """
    emb = ad.AnnData(
        X=np.zeros((adata.n_obs, 1), dtype=np.float32),  # placeholder
        obs=adata.obs[[BATCH_KEY, LABEL_KEY]].copy(),
    )
    for key in BENCHMARK_KEYS:
        emb.obsm[key] = np.asarray(adata.obsm[key])
    emb.write_h5ad(emb_file)


def process_subset(spec: pd.Series) -> None:
    sub_dir = ROBUST_ROOT / spec["subset_id"]
    emb_file = sub_dir / EMBEDDINGS_FILE
    if emb_file.exists():
        print(f"[exists] {spec['subset_id']} -> {emb_file} (skip integration)")
        return
    if (sub_dir / RESULTS_FILE).exists():
        # Already fully benchmarked in an earlier run; nothing to integrate.
        print(f"[done]   {spec['subset_id']} already benchmarked (skip integration)")
        return

    print(f"\n{'=' * 60}\n  Integrating {spec['subset_id']} "
          f"(N={spec['n_datasets']}, rep={spec['replicate']})\n{'=' * 60}")

    adata = sc.read_h5ad(sub_dir / "subset.h5ad")
    adata = integrate_python(adata)
    load_r_embeddings(adata, sub_dir)
    save_embeddings(adata, emb_file)
    print(f"[done] wrote embeddings -> {emb_file}")

    del adata
    gc.collect()


def main() -> None:
    manifest = pd.read_csv(MANIFEST_CSV)
    print(f"Loaded manifest with {len(manifest)} subsets from {MANIFEST_CSV}")
    for _, spec in manifest.iterrows():
        process_subset(spec)
    print("\nAll subset integrations complete. Now run 06_run_subset_benchmark.py")


if __name__ == "__main__":
    main()
