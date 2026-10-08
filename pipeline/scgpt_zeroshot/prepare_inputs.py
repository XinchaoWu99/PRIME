"""scGPT zero-shot benchmark, step 1: model-independent inputs.

    python prepare_inputs.py immune|lung

Writes out_dir/scgpt/<dataset>/
    obs.tsv           cell, batch, label; order of the source file (= order of the benchmark object)
    genes.tsv         var_id, symbol, is_hvg; all genes of the source file, HVG = the 2,000 batch-aware cell_ranger genes
                      used by every other method
    counts.npz        raw counts, cells x all genes (CSR); input of the all-genes run
    lognorm_hvg.npz   log-normalised expression of the HVG columns (CSR); input of the zero-shot embedding
    check.json        identity checks against the saved benchmark object

The two data sets are read exactly as in the benchmarks of the other methods (config.yaml datasets.immune /
datasets.lung): human immune from the log-normalised X of the scIB benchmark file, lung adenocarcinoma from the count
layer of the NSCLC atlas (normalise to 1e4, log1p). scGPT 0.2.4 pins an anndata version that cannot read the atlas
file, so this step runs in the main environment and hands over plain .npz / .tsv files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import common as C

SYMBOL_COL = {"immune": None, "lung": "feature_name"}     # None = var index already holds gene symbols


def h5_index(path: str, group: str = "obs") -> np.ndarray:
    with h5py.File(path, "r") as f:
        g = f[group]
        return np.array([x.decode() if isinstance(x, bytes) else x for x in g[g.attrs["_index"]][:]])


def saved_hvg(bench: str, var_ids: np.ndarray) -> np.ndarray | None:
    """HVG flags stored with the benchmark object (immune: var.highly_variable; lung: X = the 2,000 HVGs)."""
    path = bench
    with h5py.File(path, "r") as f:
        var = f["var"]
        ids = np.array([x.decode() if isinstance(x, bytes) else x for x in var[var.attrs["_index"]][:]])
        if "highly_variable" in var:
            return np.isin(var_ids, ids[var["highly_variable"][:].astype(bool)])
        if len(ids) == 2000:
            return np.isin(var_ids, ids)
    return None


def main(name: str) -> None:
    cfg = C.load_config()
    d = cfg["datasets"][name]
    out = Path(cfg["out_dir"]) / "scgpt" / name
    out.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(d["path"], backed="r" if d.get("subset") else None)
    for key, value in (d.get("subset") or {}).items():
        adata = adata[adata.obs[key] == value]
    adata = adata.to_memory() if adata.isbacked or d.get("subset") else adata
    print(f"{name}: {adata.n_obs} cells x {adata.n_vars} genes", flush=True)

    obs = pd.DataFrame({"batch": adata.obs[d["batch"]].astype(str).values,
                        "label": adata.obs[d["label"]].astype(str).values}, index=adata.obs_names.astype(str))
    counts = adata.layers[d.get("counts_layer", "counts")].copy()
    counts = sp.csr_matrix(counts, dtype=np.float32)
    symbols = adata.var_names.astype(str) if SYMBOL_COL[name] is None else adata.var[SYMBOL_COL[name]].astype(str)
    var_ids = adata.var_names.astype(str).values

    # expression exactly as fed to every other method
    if d["input"] == "counts":
        adata.X = counts.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    adata.obs["batch"] = obs["batch"].values
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    sc.pp.highly_variable_genes(adata, batch_key="batch", **d["hvg"])
    hvg = adata.var["highly_variable"].to_numpy()
    X = sp.csr_matrix(adata.X, dtype=np.float32)

    check = {"n_obs": int(adata.n_obs), "n_genes": int(adata.n_vars), "n_hvg": int(hvg.sum()),
             "n_batches": int(obs["batch"].nunique()), "n_labels": int(obs["label"].nunique())}
    bench = cfg["benchmark"][name]["embeddings_h5ad"]
    check["cell_order_equals_benchmark"] = bool(np.array_equal(h5_index(bench), obs.index.values))
    ref = saved_hvg(bench, var_ids)
    check["hvg_equals_benchmark"] = None if ref is None else bool(np.array_equal(ref, hvg))
    print(check, flush=True)

    obs.to_csv(out / "obs.tsv", sep="\t")
    pd.DataFrame({"var_id": var_ids, "symbol": symbols.values, "is_hvg": hvg}).to_csv(out / "genes.tsv", sep="\t", index=False)
    sp.save_npz(out / "counts.npz", counts, compressed=False)
    sp.save_npz(out / "lognorm_hvg.npz", X[:, np.flatnonzero(hvg)].tocsr(), compressed=False)
    with open(out / "check.json", "w") as f:
        json.dump(check, f, indent=1)
    assert check["cell_order_equals_benchmark"], "cell order differs from the benchmark object"


if __name__ == "__main__":
    main(sys.argv[1])
