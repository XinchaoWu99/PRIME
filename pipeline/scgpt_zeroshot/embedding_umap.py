"""scGPT zero-shot benchmark: UMAP coordinates of the scGPT embeddings, with the settings used for the other methods
of the same data set.

    python embedding_umap.py DATASET NAME=embedding.npy [NAME=embedding.npy ...]

immune: 15 neighbours on all embedding dimensions, UMAP min_dist 0.3, spread 1.0 (benchmark/immune/integrate_and_benchmark.py);
lung:   scanpy defaults (benchmark/nsclc/02_run_benchmark.py).
Output: out_dir/scgpt/<dataset>/umap/<NAME>.npy  (cells x 2, order of obs.tsv)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import common as C

UMAP_KW = {"immune": {"min_dist": 0.3, "spread": 1.0}, "lung": {}}

if __name__ == "__main__":
    dataset, specs = sys.argv[1], sys.argv[2:]
    cfg = C.load_config()
    out = Path(cfg["out_dir"]) / "scgpt" / dataset / "umap"
    out.mkdir(parents=True, exist_ok=True)
    sc.settings.n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    for spec in specs:
        name, path = spec.split("=", 1)
        X = np.load(path)
        adata = sc.AnnData(X=np.zeros((X.shape[0], 1), dtype=np.float32), obsm={"emb": X})
        sc.pp.neighbors(adata, use_rep="emb")
        sc.tl.umap(adata, **UMAP_KW[dataset])
        np.save(out / f"{name}.npy", adata.obsm["X_umap"])
        print("wrote", out / f"{name}.npy", flush=True)
