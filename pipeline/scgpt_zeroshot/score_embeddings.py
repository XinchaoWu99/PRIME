"""scGPT zero-shot benchmark, step 3: scIB scores under the protocol of the benchmark tables.

    python score_embeddings.py DATASET NAME=embedding.npy [NAME=embedding.npy ...] [--ref Unintegrated Harmony ...] [--tag TAG]

Scores the given embeddings exactly as the other methods were scored (benchmark/immune/integrate_and_benchmark.py,
benchmark/nsclc/02_run_benchmark.py; scib-metrics 0.5.9):
  Benchmarker(batch_key = batch, label_key = cell type, Leiden NMI + ARI, cLISI, BRAS, iLISI, kBET, PCR comparison,
              no silhouette-label, no graph connectivity), the pre-integrated embedding for the PCR comparison = the default
              of the Benchmarker (PCA of the saved adata.X), scores not min-max scaled;
  isolated-label silhouette added separately, aggregate scores recomputed as in the benchmark tables
  (Bio = mean(NMI, ARI, cLISI, isolated), Batch = mean(BRAS, iLISI, kBET, PCR), Total = 0.6 Bio + 0.4 Batch).
adata.X, the batch / label columns and the reference embeddings are read from the saved benchmark object
(config.yaml benchmark.<dataset>.embeddings_h5ad), so the new rows and the existing rows come from the same cells.
Every method is scored independently of the others (neighbours are computed per embedding), so adding rows does not
change existing ones. `--ref` names methods of the benchmark object that are re-scored here as a reproduction check
against the saved table.

Output: out_dir/scgpt/<dataset>/scores/<tag>.csv (same layout as benchmark_results_with_isolated_labels.csv, only the
rows scored here) and <tag>_vs_benchmark.csv (re-scored reference rows next to the saved values).
"""
from __future__ import annotations

import argparse
import os
import time
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import anndata as ad
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import common as C

ISO, BIO, BATCH, TOTAL, METRIC_TYPE, AGG = "Isolated labels", "Bio conservation", "Batch correction", "Total", "Metric Type", "Aggregate score"


def isolated_labels(labels: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Labels present in the fewest batches (as `prime.metrics` `_get_isolated_labels`, iso_threshold = the minimum)."""
    tmp = pd.DataFrame({"label": labels, "batch": batch}).drop_duplicates()
    per_label = tmp.groupby("label")["batch"].count()
    return per_label[per_label <= int(per_label.min())].index.to_numpy()


def isolated_label_score(X: np.ndarray, labels: np.ndarray, batch: np.ndarray, chunk: int = 2000) -> float:
    """Mean silhouette (rescaled to 0-1) of the isolated labels' cells.

    Same value as `prime.metrics.isolated_label_score_single` (sklearn silhouette_samples of all cells, then the mean
    per isolated label, then the mean over labels), but the silhouette is only evaluated for the rows that are used
    (cells of isolated labels) against all cells, which is exact and avoids the all-against-all distance matrix.
    """
    iso = isolated_labels(labels, batch)
    codes, uniq = pd.factorize(labels)
    n_lab = len(uniq)
    counts = np.bincount(codes, minlength=n_lab).astype(np.float64)
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    onehot = torch.zeros(len(codes), n_lab)
    onehot[torch.arange(len(codes)), torch.from_numpy(codes)] = 1.0
    scores = []
    for lab in iso:
        c = int(np.flatnonzero(uniq == lab)[0])
        rows = np.flatnonzero(codes == c)
        s = np.zeros(len(rows))
        for a in range(0, len(rows), chunk):
            r = torch.from_numpy(rows[a:a + chunk])
            D = torch.cdist(Xt[r].double(), Xt.double()) if len(Xt) < 50000 else torch.cdist(Xt[r], Xt)
            S = (D @ onehot.to(D.dtype)).double().numpy()                    # summed distance to every label
            own = S[:, c] / max(counts[c] - 1, 1)
            other = S / counts[None, :]
            other[:, c] = np.inf
            b = other.min(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                si = (b - own) / np.maximum(own, b)
            si[~np.isfinite(si)] = 0.0
            if counts[c] <= 1:
                si[:] = 0.0
            s[a:a + chunk] = si
        scores.append(float(np.mean((s + 1.0) / 2.0)))
    return float(np.mean(scores))


def score(dataset: str, embeddings: dict[str, np.ndarray], refs: list[str], n_jobs: int) -> pd.DataFrame:
    from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
    cfg = C.load_config()
    src = cfg["benchmark"][dataset]["embeddings_h5ad"]
    t0 = time.time()
    bench = ad.read_h5ad(src, backed="r")
    X = bench.X[:]
    obs = bench.obs[["batch", "cell_type"]].copy()
    obs["batch"] = obs["batch"].astype(str).astype("category")
    obs["cell_type"] = obs["cell_type"].astype(str).astype("category")
    in_obs = pd.read_csv(Path(cfg["out_dir"]) / "scgpt" / dataset / "obs.tsv", sep="\t", index_col=0)
    assert (in_obs.index.values == obs.index.astype(str).values).all(), "cell order differs from the benchmark object"
    assert (in_obs["batch"].values == obs["batch"].astype(str).values).all()
    assert (in_obs["label"].values == obs["cell_type"].astype(str).values).all()
    adata = ad.AnnData(X=X, obs=obs)
    keys = list(embeddings) + refs
    for k in refs:
        adata.obsm[k] = np.asarray(bench.obsm[k])
    for k, v in embeddings.items():
        assert v.shape[0] == adata.n_obs and np.isfinite(v).all(), k
        adata.obsm[k] = v
    print(f"{dataset}: {adata.n_obs} cells, X {X.shape}, scoring {keys} ({time.time() - t0:.0f} s)", flush=True)

    bm = Benchmarker(
        adata, batch_key="batch", label_key="cell_type",
        bio_conservation_metrics=BioConservation(nmi_ari_cluster_labels_leiden=True, nmi_ari_cluster_labels_kmeans=False,
                                                 silhouette_label=False, isolated_labels=False),
        batch_correction_metrics=BatchCorrection(graph_connectivity=False),
        embedding_obsm_keys=keys, n_jobs=n_jobs, progress_bar=False)
    bm.benchmark()
    res = bm.get_results(min_max_scale=False)
    print(f"scIB done ({time.time() - t0:.0f} s)", flush=True)

    labels, batch = obs["cell_type"].astype(str).to_numpy(), obs["batch"].astype(str).to_numpy()
    iso = {k: isolated_label_score(np.asarray(adata.obsm[k]), labels, batch) for k in keys}
    print(f"isolated labels: {iso} ({time.time() - t0:.0f} s)", flush=True)

    # table layout: Isolated labels sits before the aggregate columns; aggregates recomputed with it
    out = res.copy()
    out[ISO] = np.nan
    methods = out.index[out.index != METRIC_TYPE]
    out.loc[methods, ISO] = [iso[m] for m in methods]
    out.loc[METRIC_TYPE, ISO] = BIO
    cols = [c for c in out.columns if c != ISO]
    first_agg = [c for c in cols if out.loc[METRIC_TYPE, c] == AGG][0]
    at = cols.index(first_agg)
    out = out[cols[:at] + [ISO] + cols[at:]]
    bio_cols = [c for c in out.columns if out.loc[METRIC_TYPE, c] == BIO and c not in (BIO,)]
    out.loc[methods, BIO] = out.loc[methods, bio_cols].astype(float).mean(axis=1)
    out.loc[methods, TOTAL] = 0.6 * out.loc[methods, BIO].astype(float) + 0.4 * out.loc[methods, BATCH].astype(float)
    for c in (BATCH, BIO, TOTAL):
        out.loc[METRIC_TYPE, c] = AGG
    out.index.name = "Embedding"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["immune", "lung"])
    ap.add_argument("embeddings", nargs="+", help="NAME=path.npy")
    ap.add_argument("--ref", nargs="*", default=[], help="methods of the benchmark object to re-score as a reproduction check")
    ap.add_argument("--tag", default="scgpt")
    ap.add_argument("--verify-isolated", action="store_true",
                    help="check the isolated-label score against sklearn silhouette_samples (small data sets only)")
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    cfg = C.load_config()
    emb = {}
    for spec in a.embeddings:
        k, p = spec.split("=", 1)
        emb[k] = np.load(p)
    if a.verify_isolated:
        from sklearn.metrics import silhouette_samples
        b = ad.read_h5ad(cfg["benchmark"][a.dataset]["embeddings_h5ad"], backed="r")
        lab, bat = b.obs["cell_type"].astype(str).to_numpy(), b.obs["batch"].astype(str).to_numpy()
        for k, X in [*emb.items(), *[(r, np.asarray(b.obsm[r])) for r in a.ref[:1]]]:
            sil = (silhouette_samples(X, lab, metric="euclidean") + 1) / 2
            ref = float(np.mean([sil[lab == l].mean() for l in isolated_labels(lab, bat)]))
            print(f"isolated-label check {k}: exact {ref:.6f}  restricted {isolated_label_score(X, lab, bat):.6f}", flush=True)
    out = score(a.dataset, emb, a.ref, a.n_jobs)
    d = Path(cfg["out_dir"]) / "scgpt" / a.dataset / "scores"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / f"{a.tag}.csv")
    print(out.round(4).to_string())
    if a.ref:
        man = pd.read_csv(cfg["benchmark"][a.dataset]["results_csv"], index_col=0)
        rows = []
        for k in a.ref:
            for col in out.columns:
                if col in man.columns and k in man.index:
                    rows.append({"method": k, "metric": col, "saved": float(man.loc[k, col]), "rescored": float(out.loc[k, col])})
        cmp_ = pd.DataFrame(rows)
        cmp_["difference"] = cmp_["rescored"] - cmp_["saved"]
        cmp_.to_csv(d / f"{a.tag}_vs_benchmark.csv", index=False)
        print(cmp_.round(4).to_string())
