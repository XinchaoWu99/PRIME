"""Parameter sensitivity, seed stability and module ablation of PRIME.

    python parameter_grid.py DATASET {sensitivity,ablation} [--task T --ntasks N]

`sensitivity` runs the one-factor and interaction grids of config.yaml (`grids`) around the dataset's default
parameters; `ablation` runs the named ablations (`ablations`). Configurations that PRIME cannot distinguish
(same K and same minimum vote count) are run once. For scRNA-seq data (mode `prime`) the K_max projection votes
are computed once per (seed, d, k) and every (K, tau) is derived from them; `prime_st` is rerun per configuration.
With --ntasks N the work units are split over N tasks (--task, default SLURM_ARRAY_TASK_ID).

Outputs (out_dir/<grid>/DATASET):
  results_task<T>.tsv     one row per configuration x seed: anchor metrics, integration metrics (+ XLC for
                          spatial data), runtime, seed stability (Jaccard of anchor sets and of 15-NN sets)
  embeddings/<name>_s<seed>.npy   the default configuration (sensitivity) / every ablation variant (ablation)
  unintegrated.tsv        the same integration metrics on the unintegrated PCA (scRNA-seq data, sensitivity, task 0)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import anchors as A
import common as C
from prime import prime_st  # noqa: E402
from prime.metrics.xlc import ordinal_layer_continuity  # noqa: E402


def expand(grid: str, mode: str, cfg: dict) -> list[tuple[str, dict]]:
    if grid == "ablation":
        return list(cfg["ablations"][mode].items())
    g = cfg["grids"][mode]
    out = [("default", {})]
    for p, vals in g["one_factor"].items():
        out += [(f"{p}={v}", {p: v}) for v in vals]
    for inter in g["interactions"]:
        (p1, v1), (p2, v2) = inter.items()
        out += [(f"{p1}={a},{p2}={b}", {p1: a, p2: b}) for a in v1 for b in v2]
    return out


def canonical(params: dict, mode: str) -> tuple:
    """Configurations with equal key give identical PRIME output."""
    K = params["n_projections"]
    p = dict(params, consensus_threshold=A.min_votes(K, params["consensus_threshold"]))
    p.setdefault("smoothing", True)
    return tuple(sorted(p.items()))


def knn_sets(Z: np.ndarray, k: int = 15) -> np.ndarray:
    return NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]


def seed_stability(anchor_keys: dict, views: dict) -> dict:
    """Mean pairwise Jaccard across seeds: anchor sets (pairing) and 15-NN sets (function)."""
    seeds = sorted(anchor_keys)
    if len(seeds) < 2:
        return {}
    jac_a, jac_n = [], []
    nn = {s: knn_sets(views[s]) for s in seeds}
    for s, t in combinations(seeds, 2):
        a, b = anchor_keys[s], anchor_keys[t]
        jac_a.append(len(np.intersect1d(a, b)) / max(len(np.union1d(a, b)), 1))
        inter = np.array([len(np.intersect1d(x, y)) for x, y in zip(nn[s], nn[t])])
        jac_n.append(np.mean(inter / (2 * nn[s].shape[1] - inter)))
    return {"anchor_jaccard": np.mean(jac_a), "knn15_jaccard": np.mean(jac_n)}


def score(adata, Z, ds_cfg, cfg, seed) -> dict:
    """Integration metrics (+ XLC for spatial data) of one embedding."""
    adata.obsm["PRIME"] = C.eval_view(Z, cfg["eval_pcs"], seed)
    out = C.integration_metrics(adata, ["PRIME"]).loc["PRIME"].to_dict()
    if ds_cfg["mode"] == "prime_st":
        out["XLC"] = ordinal_layer_continuity(adata.obsm["PRIME"], adata.obs["label"].astype(str).values,
                                              slices=adata.obs["batch"].values, n_perm=100)
    return out


def run_prime(adata, units, ds_cfg, cfg, seeds, save_names):
    """Non-spatial PRIME: one vote matrix per (seed, d, k), every (K, tau, sigma) from it."""
    X, batch, lab = adata.X, adata.obs["batch"].values, adata.obs["label"].values
    rows, stab = [], defaultdict(lambda: ({}, {}))
    for (d, k), members in units:
        K_max = max(p["n_projections"] for _, p in members)
        for seed in seeds:
            votes = A.projection_votes(X, batch, K_max, d, k, seed)
            for names, p in members:
                K = p["n_projections"]
                i, j, w = A.consensus(votes, K, p["consensus_threshold"])
                with C.ResourceMeter() as meter:
                    Z = A.correct(adata, i, j, w, p["sigma"], p.get("smoothing", True))
                am, _ = A.anchor_metrics(i, j, lab, batch)
                rows.append({"names": names, "seed": seed, "params": json.dumps(p), **am,
                             "seconds": votes["seconds"][:K].sum() + meter.seconds,
                             "peak_gb_correction": meter.peak_gb,
                             **score(adata, Z, ds_cfg, cfg, seed)})
                stab[names][0][seed] = i * votes["n"] + j
                stab[names][1][seed] = adata.obsm["PRIME"]
                if names.split(";")[0] in save_names:
                    np.save(save_names[names.split(";")[0]].format(seed=seed), adata.obsm["PRIME"])
    return rows, stab


def run_prime_st(adata, units, ds_cfg, cfg, seeds, save_names):
    rows, stab = [], defaultdict(lambda: ({}, {}))
    batch, lab = adata.obs["batch"].values, adata.obs["label"].values
    for names, p in units:
        for seed in seeds:
            ad = adata.copy()
            with C.ResourceMeter() as meter:
                prime_st(ad, batch_key="batch", random_state=seed, store_graphs=True, verbose=False, **p)
            g = ad.uns["prime_graphs"]
            i, j = (x.astype(np.int64) for x in g["W_anchor"].nonzero())
            i, j = i[i < j], j[i < j]
            am, _ = A.anchor_metrics(i, j, lab, batch, strategy=p.get("mnn_strategy", "star"))
            rows.append({"names": names, "seed": seed, "params": json.dumps(p), **am,
                         "seconds": meter.seconds, "peak_gb": meter.peak_gb,
                         "cg_failed_dims": g["cg_failed_dims"],
                         **score(adata, ad.obsm["X_prime"], ds_cfg, cfg, seed)})
            stab[names][0][seed] = i * ad.n_obs + j
            stab[names][1][seed] = adata.obsm["PRIME"]
            if names.split(";")[0] in save_names:
                np.save(save_names[names.split(";")[0]].format(seed=seed), ad.obsm["X_prime"])
    return rows, stab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("grid", choices=["sensitivity", "ablation"])
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--ntasks", type=int, default=1)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    ds_cfg = cfg["datasets"][args.dataset]
    mode, seeds = ds_cfg["mode"], ds_cfg.get("seeds", cfg["seeds"])
    out = Path(cfg["out_dir"]) / args.grid / args.dataset
    (out / "embeddings").mkdir(parents=True, exist_ok=True)

    uniq = {}
    for name, over in expand(args.grid, mode, cfg):
        p = {**ds_cfg["default"], **over}
        uniq.setdefault(canonical(p, mode), [[], p])[0].append(name)
    configs = [(";".join(names), p) for names, p in uniq.values()]
    save_names = {n: str(out / "embeddings" / (n.replace("=", "") + "_s{seed}.npy"))
                  for n in (["default"] if args.grid == "sensitivity" else cfg["ablations"][mode])}

    if mode == "prime":   # work unit = all configs sharing (d, k), so votes are reused
        groups = defaultdict(list)
        for names, p in configs:
            groups[(p["target_dim"], p["k_neighbors"])].append((names, p))
        units = sorted(groups.items())
    else:
        units = configs
    units = units[args.task::args.ntasks]

    adata = C.load_dataset(args.dataset, cfg)
    if mode == "prime" and args.grid == "sensitivity" and args.task == 0:
        # reference of the sensitivity figure: the same metric set on the unintegrated PCA
        C.integration_metrics(adata, ["Unintegrated"]).to_csv(out / "unintegrated.tsv", sep="\t")
    runner = run_prime if mode == "prime" else run_prime_st
    rows, stab = runner(adata, units, ds_cfg, cfg, seeds, save_names)

    res = pd.DataFrame(rows)
    stab_rows = pd.DataFrame([{"names": n, **seed_stability(*v)} for n, v in stab.items()])
    res = res.merge(stab_rows, on="names", how="left")
    C.append_rows(res, out / f"results_task{args.task}.tsv")


if __name__ == "__main__":
    main()
