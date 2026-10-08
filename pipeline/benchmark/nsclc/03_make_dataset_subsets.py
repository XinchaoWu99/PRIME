"""Robustness experiment, step 1/3 — build random dataset subsets.

The lung-adenocarcinoma slice of ``NSCLC_atlas.h5ad`` is composed of 12
constituent datasets (the ``dataset`` / ``batch`` column).  To demonstrate that
the integration is robust to *how many* datasets are jointly integrated, we draw
random subsets of N = 4, 6, 8, 10 datasets (5 independent replicates each) and
save one raw-counts AnnData per subset.

Each saved ``subset.h5ad`` contains the *full* gene set and a raw ``layers/count``
layer so that, downstream, both the Python pipeline (script 05) and the R
pipeline (script 04) can recompute HVGs / normalization from scratch — exactly
as script 01 and ../r_methods/run_r_methods.R do for the full 12-dataset data.

Run order:  03 (this)  ->  04 (R-based tools)  ->  05 (Python integration)  ->  06 (benchmark).

Usage:
    python 03_make_dataset_subsets.py
"""
from __future__ import annotations

import gc
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # pipeline/ (common.py)
import common as C  # noqa: E402

CFG = C.load_config()

# ============================================================================
# User settings (edit these first)
# ============================================================================
ATLAS_FILE = Path(CFG["datasets"]["lung"]["path"])
DISEASE_VALUE = "lung adenocarcinoma"      # selects the 12-dataset lung_cancer slice
DATASET_COL = "dataset"                     # column that defines a "dataset"
LABEL_SRC_COL = "ann_fine"                  # -> adata.obs["cell_type"]
COUNTS_LAYER = "count"                       # raw counts layer in the atlas

# Output root (next to the benchmark outputs of the full data)
OUTPUT_ROOT = Path(CFG["benchmark"]["lung"]["dir"]) / "robustness"

# Subsetting design
SUBSET_SIZES = [4, 6, 8, 10]
N_REPLICATES = 5
MASTER_SEED = 0

# ============================================================================


def _seed_for(n_datasets: int, replicate: int) -> int:
    """Deterministic, collision-free seed per (size, replicate)."""
    return MASTER_SEED * 1_000_000 + n_datasets * 1_000 + replicate


def choose_subsets(all_datasets: list[str]) -> list[dict]:
    """Pick N_REPLICATES unique dataset subsets for each size in SUBSET_SIZES.

    Uniqueness is enforced *within* each size; if the requested number of
    replicates exceeds the number of possible combinations, it is capped.
    """
    plan: list[dict] = []
    all_datasets = sorted(all_datasets)

    for n in SUBSET_SIZES:
        if n > len(all_datasets):
            print(f"  [skip] N={n} > available datasets ({len(all_datasets)})")
            continue

        max_combos = len(list(combinations(all_datasets, n)))
        n_reps = min(N_REPLICATES, max_combos)
        if n_reps < N_REPLICATES:
            print(f"  [warn] N={n}: only {max_combos} unique subsets exist; "
                  f"using {n_reps} replicates instead of {N_REPLICATES}.")

        seen: set[frozenset] = set()
        rep = 0
        attempts = 0
        while rep < n_reps:
            seed = _seed_for(n, rep + 1) + attempts * 7919
            rng = np.random.default_rng(seed)
            chosen = frozenset(rng.choice(all_datasets, size=n, replace=False).tolist())
            attempts += 1
            if chosen in seen:
                continue
            seen.add(chosen)
            rep += 1
            plan.append({
                "subset_id": f"n{n:02d}_rep{rep}",
                "n_datasets": n,
                "replicate": rep,
                "seed": seed,
                "datasets": sorted(chosen),
            })
    return plan


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Reading atlas (backed): {ATLAS_FILE}")
    atlas = sc.read_h5ad(ATLAS_FILE, backed="r")
    print(f"  full atlas: {atlas.shape[0]} cells x {atlas.shape[1]} genes")

    mask = atlas.obs["disease"].astype(str) == DISEASE_VALUE
    print(f"  selecting disease == {DISEASE_VALUE!r}: {int(mask.sum())} cells")
    adata = atlas[mask.values].to_memory()
    del atlas
    gc.collect()

    # Mirror script 01: batch = dataset, cell_type = ann_fine, X = raw counts.
    adata.obs["batch"] = adata.obs[DATASET_COL].astype("category")
    adata.obs["cell_type"] = adata.obs[LABEL_SRC_COL].astype("category")
    if COUNTS_LAYER not in adata.layers:
        raise ValueError(f"Expected raw counts in adata.layers[{COUNTS_LAYER!r}]")
    adata.X = adata.layers[COUNTS_LAYER].copy()

    # Slim down to what the downstream pipelines actually need.
    for attr in ("obsm", "varm", "obsp", "varp"):
        getattr(adata, attr).clear()
    adata.uns = {}
    if adata.raw is not None:
        adata.raw = None

    all_datasets = adata.obs["batch"].cat.categories.tolist()
    print(f"  found {len(all_datasets)} datasets: {all_datasets}")

    plan = choose_subsets(all_datasets)
    print(f"\nPlanned {len(plan)} subsets "
          f"({SUBSET_SIZES} x up to {N_REPLICATES} replicates).")

    manifest_rows = []
    for spec in plan:
        sub_dir = OUTPUT_ROOT / spec["subset_id"]
        sub_dir.mkdir(parents=True, exist_ok=True)
        out_file = sub_dir / "subset.h5ad"

        sel = adata.obs["batch"].isin(spec["datasets"]).values
        sub = adata[sel].copy()
        # Drop now-empty categories so HVG(batch_key) and R's split() behave.
        sub.obs["batch"] = sub.obs["batch"].cat.remove_unused_categories()
        sub.obs["cell_type"] = sub.obs["cell_type"].cat.remove_unused_categories()
        sub.layers["count"] = sub.X.copy()

        row = {
            "subset_id": spec["subset_id"],
            "n_datasets": spec["n_datasets"],
            "replicate": spec["replicate"],
            "seed": spec["seed"],
            "n_cells": int(sub.n_obs),
            "n_celltypes": int(sub.obs["cell_type"].nunique()),
            "datasets": ";".join(spec["datasets"]),
            "h5ad": str(out_file),
        }
        manifest_rows.append(row)

        if out_file.exists():
            print(f"  [exists] {spec['subset_id']} -> {out_file} (skip write)")
        else:
            print(f"  [write]  {spec['subset_id']}: {sub.n_obs} cells, "
                  f"{spec['n_datasets']} datasets -> {out_file}")
            sub.write_h5ad(out_file)

        del sub
        gc.collect()

    manifest = pd.DataFrame(manifest_rows)
    manifest_file = OUTPUT_ROOT / "subset_manifest.csv"
    manifest.to_csv(manifest_file, index=False)
    print(f"\nWrote manifest: {manifest_file}")
    print(manifest[["subset_id", "n_datasets", "replicate", "n_cells", "datasets"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
