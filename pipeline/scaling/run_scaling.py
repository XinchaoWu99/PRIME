"""Runtime and peak memory against the number of cells (N) and the number of batches (B).

    python run_scaling.py prepare                      # nested stratified subsets of scaling.dataset
    python run_scaling.py run METHOD SUBSET.h5ad [--threads 16] [--prime-chunk 2000] [--rep 1]
    python run_scaling.py quality [SUBSET]            # integration metrics of every finished run

METHOD is prime / harmony / scanorama / scvi (Python) or fastmnn / rpca (R, r_methods.R).

`run` executes the method in a fresh child process and meters it from outside: wall time, peak RSS of the whole
process tree (summed at the same instant) and peak GPU memory of those processes. The child reports its own
stage times (shared preprocessing vs integration core). OOM / timeout / errors are recorded as such, never
dropped. --prime-chunk sets the number of genes PRIME corrects at a time (`chunk_size` of
`prime.ensemble_mnn_correct`; smaller = less memory, identical output).
Results: out_dir/scaling/subsets/ (subsets and manifest.tsv), out_dir/scaling/runs/<method>_<subset>.json and
.npy, out_dir/scaling/quality.tsv.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import common as C


def prepare(cfg):
    sc_cfg, out = cfg["scaling"], Path(cfg["out_dir"]) / "scaling" / "subsets"
    out.mkdir(parents=True, exist_ok=True)
    d = cfg["datasets"][sc_cfg["dataset"]]
    import scanpy as sc
    adata = sc.read_h5ad(d["path"], backed="r")
    for key, value in (d.get("subset") or {}).items():
        adata = adata[adata.obs[key] == value]
    adata = adata.to_memory()
    adata.X = adata.layers[d["counts_layer"]]
    adata.layers.clear()
    adata.obs = pd.DataFrame({"batch": adata.obs[d["batch"]].astype(str),
                              "label": adata.obs[d["label"]].astype(str)}, index=adata.obs_names)
    order = C.stratified_order(adata.obs)
    manifest = []
    for n in sc_cfg["n_cells"] + [adata.n_obs]:
        sub = adata[np.sort(order[:min(n, adata.n_obs)])].copy()
        name = f"N{sub.n_obs}"
        sub.write_h5ad(out / f"{name}.h5ad")
        manifest.append({"subset": name, "axis": "N", "n_cells": sub.n_obs, "n_batches": sub.obs["batch"].nunique()})

    n_tot = sc_cfg["n_cells_batch_axis"]
    sizes = adata.obs["batch"].value_counts()
    ds_order = list(np.random.default_rng(0).permutation(sorted(sizes.index)))
    for b in sc_cfg["n_batches"]:
        per = n_tot // b
        chosen = [x for x in ds_order if sizes[x] >= per][:b]
        if len(chosen) < b:
            manifest.append({"subset": f"B{b}", "axis": "B", "status": f"only {len(chosen)} batches with >= {per} cells"})
            continue
        idx = np.concatenate([
            np.flatnonzero(adata.obs["batch"].values == x)[C.stratified_order(adata.obs[adata.obs["batch"] == x])[:per]]
            for x in chosen])
        sub = adata[np.sort(idx)].copy()
        sub.write_h5ad(out / f"B{b}.h5ad")
        manifest.append({"subset": f"B{b}", "axis": "B", "n_cells": sub.n_obs, "n_batches": b, "batches": ";".join(chosen)})
    pd.DataFrame(manifest).to_csv(out / "manifest.tsv", sep="\t", index=False)


def core(method: str, subset: str, out_npy: str, prime_chunk: int, cfg: dict) -> dict:
    """Runs inside the metered child. Returns stage times."""
    import scanpy as sc
    t = {}
    t0 = time.perf_counter()
    adata = sc.read_h5ad(subset)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key="batch")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.tl.pca(adata, n_comps=30)
    t["shared_preprocess_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if method == "prime":
        from prime import ensemble_mnn_correct
        p = cfg["prime_default"]
        Z = ensemble_mnn_correct(adata, "batch", random_state=42, inplace=False, chunk_size=prime_chunk, **p)
    elif method == "harmony":
        from harmonypy import run_harmony
        Z = np.asarray(run_harmony(adata.obsm["X_pca"], adata.obs, "batch").Z_corr)
        Z = Z.T if Z.shape[0] != adata.n_obs else Z
    elif method == "scanorama":
        import scanorama
        parts = [adata[adata.obs["batch"] == b].copy() for b in adata.obs["batch"].unique()]
        scanorama.integrate_scanpy(parts, dimred=100)
        Z = pd.concat([pd.DataFrame(a.obsm["X_scanorama"], index=a.obs_names) for a in parts]).loc[adata.obs_names].values
    elif method == "scvi":
        import scvi
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        model = scvi.model.SCVI(adata)
        model.train(max_epochs=int(min(round(20000 / adata.n_obs * 400), 400)))
        Z = model.get_latent_representation()
    else:
        raise ValueError(method)
    t["core_s"] = time.perf_counter() - t0
    np.save(out_npy, np.asarray(Z, dtype=np.float32))
    return t


def tree_usage(pid: int) -> tuple[int, int]:
    """(RSS bytes, GPU MiB) of a process and all its descendants, at one instant."""
    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0, 0
    pids, rss = {p.pid for p in procs}, 0
    for p in procs:
        try:
            rss += p.memory_info().rss
        except psutil.Error:
            pass
    gpu = 0
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout
        gpu = sum(int(m) for p, m in (l.split(",") for l in q.strip().splitlines() if l) if int(p) in pids)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return rss, gpu


R_METHODS = ("fastmnn", "rpca")


def quality(cfg, only_subset=None, n_eval=50000):
    """Fixed scIB metric set on a stratified 50k-cell sample (same cells for every method
    of a subset), for every successful first run, plus the unintegrated PCA of that sample."""
    import anndata as ad
    import scanpy as sc
    root = Path(cfg["out_dir"]) / "scaling"
    runs = [json.loads(f.read_text()) | {"file": f} for f in sorted((root / "runs").glob("*.json"))
            if not f.name.endswith(".stages.json")]
    rows = []
    qf = root / "quality.tsv"
    done = set(map(tuple, pd.read_csv(qf, sep="\t")[["subset", "run"]].values)) if qf.exists() else set()
    for subset in sorted({r["subset"] for r in runs}):
        todo = [r for r in runs if r["subset"] == subset and not r.get("rep", 0) and r["status"] == "ok"
                and (subset, r["file"].stem) not in done]
        if (only_subset and subset != only_subset) or not todo:
            continue
        a = ad.read_h5ad(root / "subsets" / f"{subset}.h5ad", backed="r")
        obs = a.obs[["batch", "label"]].astype(str)
        idx = np.sort(C.stratified_order(obs, seed=1)[:n_eval])
        ref = a[idx].to_memory()
        sc.pp.normalize_total(ref, target_sum=1e4)
        sc.pp.log1p(ref)
        sc.pp.highly_variable_genes(ref, n_top_genes=2000, flavor="cell_ranger", batch_key="batch")
        ev = sc.AnnData(obs=obs.iloc[idx].astype("category"),
                        obsm={"Unintegrated": sc.pp.pca(ref[:, ref.var["highly_variable"]].X, n_comps=30)})
        keys = [] if (subset, "Unintegrated") in done else ["Unintegrated"]
        for r in todo:
            npy = r["file"].with_suffix(".npy")
            if not npy.exists():
                continue
            key = r["file"].stem
            ev.obsm[key] = C.eval_view(np.asarray(np.load(npy, mmap_mode="r")[idx]), cfg["eval_pcs"])
            keys.append(key)
        if not keys:
            continue
        res = C.integration_metrics(ev, keys, n_jobs=16)
        rows += [{"subset": subset, "run": k, "n_eval": len(idx), **res.loc[k].to_dict()} for k in keys]
        C.append_rows(rows[-len(keys):], root / "quality.tsv")


def run(args, cfg):
    out = Path(cfg["out_dir"]) / "scaling" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    name = (f"{args.method}_{Path(args.subset).stem}" + (f"_chunk{args.prime_chunk}" if args.method == "prime" else "")
            + (f"_r{args.rep}" if args.rep else ""))
    npy, stages = out / f"{name}.npy", out / f"{name}.stages.json"
    if args.method in R_METHODS:
        cmd = f"{cfg.get('rscript', 'Rscript')} {Path(__file__).resolve().parent / 'r_methods.R'} {args.subset} {npy} {stages} {args.method}"
    else:
        cmd = f"{sys.executable} {__file__} _core {args.method} {args.subset} {npy} {stages} {args.prime_chunk}"
    env = dict(os.environ, **{v: str(args.threads) for v in
                              ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")})
    t0, peak_rss, peak_gpu = time.perf_counter(), 0, 0
    proc = subprocess.Popen(cmd, shell=True, env=env)
    status = "ok"
    while proc.poll() is None:
        rss, gpu = tree_usage(proc.pid)
        peak_rss, peak_gpu = max(peak_rss, rss), max(peak_gpu, gpu)
        if time.perf_counter() - t0 > cfg["scaling"]["timeout_h"] * 3600:
            for p in psutil.Process(proc.pid).children(recursive=True) + [psutil.Process(proc.pid)]:
                p.kill()
            status = "timeout"
            break
        time.sleep(1)
    rc = proc.wait()
    if status == "ok" and rc != 0:
        status = "oom_killed" if rc in (-9, 137) else f"error_rc{rc}"
    res = {"method": args.method, "subset": Path(args.subset).stem, "status": status, "returncode": rc,
           "wall_s": time.perf_counter() - t0, "peak_rss_gb": peak_rss / 1e9, "peak_gpu_gb": peak_gpu / 1024,
           "threads": args.threads, "host": os.uname().nodename, "prime_chunk": args.prime_chunk, "rep": args.rep}
    if stages.exists():
        res.update(json.loads(stages.read_text()))
    (out / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(res)


def main():
    cfg = C.load_config()
    if sys.argv[1] == "_core":                     # child process
        _, _, method, subset, npy, stages, chunk = sys.argv
        Path(stages).write_text(json.dumps(core(method, subset, npy, int(chunk), cfg)))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "run", "quality"])
    ap.add_argument("method", nargs="?")
    ap.add_argument("subset", nargs="?")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--prime-chunk", type=int, default=2000)   # 2000 = the package default (no gene chunking at 2,000 HVGs)
    ap.add_argument("--rep", type=int, default=0)             # independent repeat index (0 = first run)
    args = ap.parse_args()
    if args.action == "quality":
        quality(cfg, args.method)                             # optional positional = subset name
    else:
        prepare(cfg) if args.action == "prepare" else run(args, cfg)


if __name__ == "__main__":
    main()
