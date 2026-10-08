"""Shared configuration, data loading, metrics and run bookkeeping of the analysis pipeline.

Every script in the sub-folders of ``pipeline/`` puts this folder on ``sys.path`` and imports this module,
which in turn makes the ``prime`` package of this checkout importable. Layout of ``pipeline/``::

    config.yaml        locations, data sets, parameter grids and method settings (the only file to edit;
                       config.local.yaml, if present, is merged over it)
    common.py          this module
    anchors.py         per-projection MNN votes, consensus anchors, anchor-level metrics
    benchmark/         integration with every method + scIB benchmark, one folder per data set
                       (r_methods: Seurat-based methods; immune; nsclc incl. robustness subsets; dlpfc)
    prime_analysis/    parameter sensitivity, seed stability, module ablation, anchor validation
    spatial/           spatial integration methods under one protocol, setting selection, DLPFC domains and markers
    visiumhd/          Visium HD mouse brain: cohort assembly and bin QC, region / layer labels, evaluation
    scaling/           runtime and memory against the number of cells and batches
    trajectory/        Monocle3 trajectories on the integrated human immune data
    scgpt_zeroshot/    scGPT (zero-shot) scored with the benchmark protocol
    slurm/run.sbatch   generic SLURM wrapper

Scripts are run from ``pipeline/`` (``python <folder>/<script> ...``); each one states its command line, inputs and
outputs in its header. Results of ``benchmark/`` stay next to the data (``benchmark.*`` of config.yaml); all other
folders write to ``out_dir/<analysis>/``, which the notebooks in ``notebooks/`` read.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import scanpy as sc
import yaml

HERE = Path(__file__).resolve().parent          # pipeline/
REPO = HERE.parent                              # repository root (the `prime` package lives here)
sys.path.insert(0, str(REPO))

_PLACEHOLDERS = ("data_root", "out_dir", "software_dir")


def _merge(base: dict, over: dict) -> dict:
    """`over` merged into `base`, dictionaries recursively."""
    out = dict(base)
    for key, value in over.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _expand(node, values: dict):
    """Replace ${data_root} / ${out_dir} / ${software_dir} in every string of the configuration."""
    if isinstance(node, dict):
        return {k: _expand(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v, values) for v in node]
    if isinstance(node, str):
        for key, value in values.items():
            node = node.replace("${" + key + "}", value)
    return node


def load_config(path: str | Path | None = None) -> dict:
    """The pipeline configuration.

    ``config.yaml`` next to this file, with ``config.local.yaml`` (if present) merged over it; ``path`` or the
    environment variable ``PRIME_CONFIG`` selects a single other file instead. Path placeholders are expanded.
    """
    path = path or os.environ.get("PRIME_CONFIG")
    if path:
        with open(path) as f:
            cfg = yaml.safe_load(f)
    else:
        with open(HERE / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        local = HERE / "config.local.yaml"
        if local.exists():
            with open(local) as f:
                cfg = _merge(cfg, yaml.safe_load(f) or {})
    return _expand(cfg, {k: str(cfg[k]) for k in _PLACEHOLDERS if k in cfg})


def load_dataset(name: str, cfg: dict) -> sc.AnnData:
    """Read a dataset and standardise obs to `batch` / `label` (and `donor`)."""
    d = cfg["datasets"][name]
    backed = "r" if d.get("subset") else None
    adata = sc.read_h5ad(d["path"], backed=backed)
    for key, value in (d.get("subset") or {}).items():
        adata = adata[adata.obs[key] == value]
    adata = adata.to_memory() if backed else adata

    adata.obs["batch"] = adata.obs[d["batch"]].astype(str).astype("category")
    adata.obs["label"] = adata.obs[d["label"]].astype(str).astype("category")
    if d.get("donor"):
        adata.obs["donor"] = adata.obs[d["donor"]].astype(str).astype("category")
    if d["mode"] == "prime_st":   # integrate all spots from raw counts; unlabelled spots ("nan") are dropped only at scoring
        return adata

    if d["input"] == "counts":
        adata.X = adata.layers[d.get("counts_layer", "counts")].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    if d.get("hvg"):
        sc.pp.highly_variable_genes(adata, batch_key="batch", **d["hvg"])
        adata = adata[:, adata.var["highly_variable"]].copy()
    sc.tl.pca(adata, n_comps=30)
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]
    return adata


def eval_view(X: np.ndarray, n_pcs: int | None, seed: int = 0) -> np.ndarray:
    """Embedding used for scoring: wide outputs (e.g. corrected genes) are reduced to top PCs."""
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    if n_pcs is None or X.shape[1] <= n_pcs:
        return X.astype(np.float32)
    return sc.pp.pca(X.astype(np.float32), n_comps=n_pcs, random_state=seed)


def integration_metrics(adata: sc.AnnData, keys: list[str], n_jobs: int = 8) -> pd.DataFrame:
    """Fixed scIB metric set (scib-metrics Benchmarker; kBET and PCR comparison off)."""
    from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

    keep = (adata.obs["label"] != "nan").to_numpy()
    obs = adata.obs.loc[keep, ["batch", "label"]].apply(lambda c: c.cat.remove_unused_categories())
    adata = sc.AnnData(obs=obs,
                       obsm={k: np.asarray(adata.obsm[k])[keep] for k in keys})
    bm = Benchmarker(
        adata, batch_key="batch", label_key="label", embedding_obsm_keys=keys,
        bio_conservation_metrics=BioConservation(isolated_labels=False),
        batch_correction_metrics=BatchCorrection(kbet_per_label=False, pcr_comparison=False),
        pre_integrated_embedding_obsm_key=keys[0],   # only used by PCR, which is off
        n_jobs=n_jobs, progress_bar=False,
    )
    bm.benchmark()
    res = bm.get_results(min_max_scale=False)
    return res.drop(index="Metric Type").astype(float)


def stratified_order(obs: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Cell order whose every prefix keeps batch x label proportions (nested subsets)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(obs))
    shuffled = obs.iloc[perm]
    pos = shuffled.groupby(["batch", "label"], observed=True).cumcount().values
    size = shuffled.groupby(["batch", "label"], observed=True)["batch"].transform("size").values
    key = (pos + rng.random(len(obs))) / size
    return perm[np.argsort(key, kind="stable")]


class ResourceMeter:
    """Wall time and peak RSS of this process tree, summed at each sampling instant."""

    def __init__(self, interval: float = 0.2):
        self.interval, self.peak, self._stop = interval, 0, threading.Event()

    def _tree_rss(self) -> int:
        proc = psutil.Process(os.getpid())
        total = 0
        for p in [proc, *proc.children(recursive=True)]:
            try:
                total += p.memory_info().rss
            except psutil.Error:
                pass
        return total

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._tree_rss())
            time.sleep(self.interval)

    def __enter__(self):
        self.t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.seconds = time.perf_counter() - self.t0
        self.peak_gb = self.peak / 1e9


def append_rows(rows: list[dict] | pd.DataFrame, path: str | Path) -> None:
    """Append result rows to a TSV, writing the header only once."""
    df = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, sep="\t")
        if list(old.columns) != list(df.columns):      # new columns: rewrite instead of misaligned append
            pd.concat([old, df]).to_csv(path, sep="\t", index=False)
            return
    df.to_csv(path, sep="\t", index=False, mode="a", header=not path.exists())
