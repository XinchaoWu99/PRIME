"""Spatial integration methods under one protocol.

    python run_methods.py METHOD SETTING [--seeds 0 1 2 3 4] [--dataset dlpfc]

METHOD is prime, staligner, graphst, precast, spabatch, spacross or spamask. Every method gets the same
sections and spots, batch = section, and raw counts (each method applies its own documented preprocessing).
SETTING names an entry of config.yaml `baselines.<METHOD>`: `default` (the official tutorial settings) or one
of the tuning settings fixed beforehand. --dataset is `dlpfc` (default) or one of the Visium HD data sets
(`brain_dg_tiles`, `brain_cortex_tiles`, `brain_sections`).

Output per seed: <SETTING>_s<seed>.npy with one row per spot of the data set in file order (NaN rows for spots
the method drops), and one row in runs.tsv (runtime, peak memory, status; failed runs are kept), written to
out_dir/spatial_methods/<METHOD>/ for DLPFC and out_dir/visiumhd/<dataset>/<METHOD>/ otherwise.

Sequence on DLPFC: every method x setting with `--seeds 0`, then select_settings.py (development donor); the default
and the selected setting with all seeds; domain_validation.py once per embedding seed (tags `emb_s<seed>` and
`tuned_emb_s<seed>`); marker_concordance.py. On the Visium HD data sets: `prime transport`, `prime unintegrated`
(the SVD reference), `precast default`, `staligner hd16`, `graphst / spacross / spabatch default`, then the scripts
of ../visiumhd/.

STAligner and GraphST are imported from their installed packages; PRECAST runs through precast.R; SpaBatch,
SpaCross and SpaMask run from their official repositories, cloned unmodified under config.yaml `software_dir`.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import traceback
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import common as C


def staligner(adata, seed, rad_cutoff=150, n_hvg=10000, margin=1.0, knn_neigh=100, n_epochs=1000):
    import scipy.sparse as sp
    import STAligner
    import torch

    parts, adjs = [], []
    for s in adata.obs["batch"].cat.categories:
        a = adata[adata.obs["batch"] == s].copy()
        STAligner.Cal_Spatial_Net(a, rad_cutoff=rad_cutoff)
        sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=n_hvg)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a = a[:, a.var["highly_variable"]].copy()
        adjs.append(a.uns["adj"])
        parts.append(a)
    cat = ad.concat(parts, join="inner")
    cat.obs["batch_name"] = cat.obs["batch"].astype(str).astype("category")   # slice, not donor
    cat.uns["edgeList"] = sp.block_diag(adjs).nonzero()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cat = STAligner.train_STAligner(cat, margin=margin, knn_neigh=knn_neigh, n_epochs=n_epochs,
                                    random_seed=seed, device=dev)
    return pd.DataFrame(cat.obsm["STAligner"], index=cat.obs_names)


def graphst(adata, seed, dim_output=64, epochs=600, **kw):
    """GraphST on concatenated slices; coordinates are offset per slice so that its
    spatial kNN graph cannot link spots of different slices."""
    import torch
    from GraphST import GraphST

    a = adata.copy()
    xy = np.asarray(a.obsm["spatial"], float).copy()
    shift = np.ptp(xy[:, 0]) + 1e4
    for n, s in enumerate(a.obs["batch"].cat.categories):
        xy[(a.obs["batch"] == s).to_numpy(), 0] += n * shift
    a.obsm["spatial"] = xy
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = GraphST.GraphST(a, device=dev, random_seed=seed, dim_output=dim_output, epochs=epochs, **kw).train()
    return pd.DataFrame(a.obsm["emb"], index=a.obs_names)


def precast(adata, seed, K=7, n_hvg=2000, q=15):
    cfg = C.load_config()
    rscript = cfg.get("rscript", "Rscript")
    if adata.obs.get("dataset", pd.Series(["dlpfc"])).iloc[0] == "dlpfc":
        data_root = str(Path(cfg["datasets"]["dlpfc"]["path"]).parent)      # the 12 Space Ranger folders
    else:                                                                    # other data: pass an h5ad
        if "array_row" not in adata.obs:     # Visium HD bin names carry the grid position: s_<size>um_<row>_<col>-1
            rc = adata.obs_names.str.extract(r"_(\d+)_(\d+)-1$").astype(int)
            adata = adata.copy()
            adata.obs["array_row"], adata.obs["array_col"] = rc[0].to_numpy(), rc[1].to_numpy()
        data_root = str(Path(cfg["out_dir"]) / "spatial_methods" / "precast" / f"input_{os.getpid()}.h5ad")
        Path(data_root).parent.mkdir(parents=True, exist_ok=True)
        obs = adata.obs[["batch", "array_row", "array_col"]].astype({"batch": str, "array_row": int, "array_col": int})
        obs.index.name = None                      # keep anndata's default "_index" key, which the R reader expects
        var = adata.var[[]].copy()
        var.index.name = None
        ad.AnnData(adata.X, obs=obs, var=var).write_h5ad(data_root)
    tmp = Path(cfg["out_dir"]) / "spatial_methods" / "precast" / f"tmp_{os.getpid()}_s{seed}.csv"   # unique per run
    tmp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([rscript, str(Path(__file__).resolve().parent / "precast.R"), data_root, str(tmp), str(seed), str(K), str(n_hvg), str(q)], check=True)
    df = pd.read_csv(tmp)
    # PRECAST appends the slice number to the barcode suffix ("...-1" -> "...-11", "...-112")
    barcode = df.pop("barcode").astype(str).str.replace(r"-\d+$", "-1", regex=True)
    section = df.pop("section").astype(str)
    df.index = barcode if data_root.endswith(".h5ad") else section + "_" + barcode   # h5ad cell names are already keys
    return df


def prime(adata, seed, **params):
    """PRIME (prime_st) through the same runner; seed s uses random_state 1000 * s (the DLPFC seeds of config.yaml)."""
    from prime import prime_st
    a = adata.copy()
    prime_st(a, batch_key="batch", random_state=1000 * seed, verbose=False, **params)
    return pd.DataFrame(a.obsm["X_prime"], index=a.obs_names)


SOFT = Path(C.load_config()["software_dir"])   # official repositories, cloned unmodified
DEVICE = "cuda:0"


def _tutorial_preprocess(a, n_hvg):
    """Per-slice preprocessing used in the SpaCross / SpaMask DLPFC tutorials."""
    a.layers["count"] = a.X.toarray()
    sc.pp.filter_genes(a, min_cells=50)
    sc.pp.filter_genes(a, min_counts=10)
    sc.pp.normalize_total(a, target_sum=1e6)
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", layer="count", n_top_genes=n_hvg)
    a = a[:, a.var["highly_variable"]].copy()
    del a.layers["count"]
    sc.pp.scale(a)
    return a


def spabatch(adata, seed, k_cutoff=8, n_hvg=3000, pre_epochs=500, epochs=1000, mask_rate=0.2):
    """All 12 slices; spatial graphs within slices, cross-slice alignment by SpaBatch's own MNN step."""
    import sys
    from sklearn.decomposition import PCA
    sys.path.insert(0, str(SOFT / "SpaBatch" / "SpaBatch"))
    from adj import combine_graph_dict, main as slice_graph
    from train import train_model
    from utils import fix_seed

    fix_seed(seed)
    parts, graph = [], None
    for s in adata.obs["batch"].cat.categories:
        a = adata[adata.obs["batch"] == s].copy()
        g = slice_graph(a, adj_cons_by="coordinate", distType="KNN", k_cutoff=k_cutoff, rad_cutoff=250)
        graph = g if graph is None else combine_graph_dict(graph, g)
        parts.append(a)
    a = ad.concat(parts)
    a.obs["batch_name"] = a.obs["batch"].astype(str)
    a.layers["count"] = a.X.toarray()
    sc.pp.filter_genes(a, min_cells=50)
    sc.pp.filter_genes(a, min_counts=10)
    sc.pp.normalize_total(a, target_sum=1e6)
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", layer="count", n_top_genes=n_hvg)
    a = a[:, a.var["highly_variable"]].copy()
    del a.layers["count"]
    sc.pp.scale(a)
    a.obsm["X_pca"] = PCA(200, random_state=42).fit_transform(a.X)
    net = train_model(a, graph, pre_epochs=pre_epochs, epochs=epochs, mask_rate=mask_rate)
    net.train_with_dec(num_aggre=1)
    feat, _ = net.process()
    return pd.DataFrame(feat, index=a.obs_names)


def spacross(adata, seed, n_hvg=5000, k_cutoff=12, topk_neighs_inter=20, stack_3d=False, model=None, train=None):
    """Default: SpaCross's batch-correction pipeline (SC_BC_pipeline) with slices placed far apart
    on z, so the spatial graph never links spots of different slices. stack_3d=True is the
    official DLPFC multi-slice notebook (ICP alignment + 3D graph, SC_pipeline) and is only
    valid for serial sections of one donor."""
    import sys
    import yaml
    from sklearn.decomposition import PCA
    import rpy2.robjects  # noqa: F401  load R from $R_HOME before SpaCross overwrites R_HOME with the authors' path
    sys.path.insert(0, str(SOFT / "SpaCross"))
    import SpaCross as T

    config = yaml.safe_load(open(SOFT / "SpaCross" / "Config" / "DLPFC.yaml"))
    config["train"]["topk_neighs_inter"] = topk_neighs_inter      # value of the official batch-integration config
    config["model"].update(model or {})                              # tuning overrides of Config/DLPFC.yaml
    config["train"].update(train or {})
    slices = list(adata.obs["batch"].cat.categories)
    parts = []
    for n, s in enumerate(slices):
        a = _tutorial_preprocess(adata[adata.obs["batch"] == s].copy(), n_hvg)
        a.obs["slice_id"] = n
        parts.append(a)
    if stack_3d:
        parts = T.align_spots(parts, method="icp")
        cat, edges = T.graph_construction3D(parts, section_ids=slices, k_cutoff=k_cutoff, rad_cutoff=None,
                                            mode="KNN", slice_dist_micron=[10] * (len(parts) - 1))
    else:
        xyz = np.concatenate([np.c_[p.obsm["spatial"], np.full(p.n_obs, n * 1e7)] for n, p in enumerate(parts)])
        cat, edges = T.graph_construction3D(parts, section_ids=slices, three_dim_coor=xyz, rad_cutoff=1.0,
                                            k_cutoff=k_cutoff, mode="KNN")
    cat.obsm["X_pca"] = PCA(200, random_state=42).fit_transform(cat.X)
    pipeline = T.SC_pipeline if stack_3d else T.SC_BC_pipeline
    net = pipeline(cat, edge_index=edges, num_clusters=7, device=DEVICE, config=config, roundseed=seed)
    net.trian()
    emb, _ = net.process()
    return pd.DataFrame(emb, index=cat.obs_names)


def spamask(adata, seed, n_hvg=5000, k_cutoff=21, max_epoch=1000, lam=2, feat_mask_rate=0.5, edge_drop_rate=0.2):
    """SpaMask integrates slices only through an ICP-aligned 3D stack, i.e. it assumes serial
    sections of one donor; config runs it with per_donor (official DLPFC setting)."""
    import sys
    import torch
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import NearestNeighbors
    import rpy2.robjects  # noqa: F401  same reason as in spacross()
    sys.path.insert(0, str(SOFT / "SpaMask"))
    sys.path.insert(0, str(SOFT / "SpaCross"))
    import SpaMask as stm
    from SpaCross.Align import align_spots        # same ICP routine as SpaMask's DLPFC script

    slices = list(adata.obs["batch"].cat.categories)
    parts = [_tutorial_preprocess(adata[adata.obs["batch"] == s].copy(), n_hvg) for s in slices]
    parts = align_spots(parts, method="icp")
    cat = ad.concat(parts)
    cat.obsm["spatial_aligned"] = np.concatenate([p.obsm["spatial_aligned"] for p in parts])
    ref = parts[0].obsm["spatial_aligned"]
    step = 10 * np.sort(np.unique(pairwise_distances(ref)))[1] / 100   # 10 um between slices, as in the script
    loc = np.c_[cat.obsm["spatial_aligned"], np.repeat(np.arange(len(parts)) * step, [p.n_obs for p in parts])]
    dist, idx = NearestNeighbors(n_neighbors=k_cutoff + 1).fit(loc).kneighbors(loc)
    net_df = pd.DataFrame({"Cell1": np.repeat(cat.obs_names, k_cutoff + 1),
                           "Cell2": cat.obs_names[idx.ravel()], "Distance": dist.ravel()})
    cat.uns["Spatial_Net"] = net_df[net_df["Distance"] > 0]
    cat.obsm["feat"] = PCA(200, random_state=42).fit_transform(cat.X)
    a = stm.utils.build_args()
    net = stm.spaMask.SPAMASK(cat, tissue_name="Donor", num_clusters=7, device=torch.device(DEVICE),
                              learning_rate=a.learning_rate, weight_decay=a.weight_decay, max_epoch=max_epoch,
                              gradient_clipping=a.gradient_clipping, feat_mask_rate=feat_mask_rate,
                              edge_drop_rate=edge_drop_rate, hidden_dim=512, latent_dim=256, bn=a.bn,
                              att_dropout_rate=a.att_dropout_rate, fc_dropout_rate=a.fc_dropout_rate,
                              use_token=a.use_token, rep_loss=a.rep_loss, rel_loss=a.rel_loss,
                              alpha=a.alpha, lam=lam, random_seed=seed, nps=a.nps)
    net.train()
    net.process(method="kmeans")
    return pd.DataFrame(net.get_adata().obsm["eval_pred"], index=cat.obs_names)


METHODS = {"prime": prime, "staligner": staligner, "graphst": graphst, "precast": precast,
           "spabatch": spabatch, "spacross": spacross, "spamask": spamask}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=list(METHODS))
    ap.add_argument("setting", help="entry of config.yaml baselines.<method>")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--dataset", default="dlpfc", help="dlpfc or a Visium HD data set of config.yaml")
    args = ap.parse_args()
    cfg = C.load_config()
    params = cfg["baselines"][args.method][args.setting] or {}
    if args.method == "prime":                       # settings override the DLPFC default parameters
        params = {**cfg["datasets"]["dlpfc"]["default"], **params}
    out = Path(cfg["out_dir"]) / ("spatial_methods" if args.dataset == "dlpfc" else f"visiumhd/{args.dataset}") / args.method
    out.mkdir(parents=True, exist_ok=True)

    adata = C.load_dataset(args.dataset, cfg)
    adata.obs["dataset"] = args.dataset
    adata.obs_names = adata.obs["key"].astype(str)      # "<slice>_<barcode>", unique
    params = dict(params)
    per_donor = params.pop("per_donor", False)     # run separately on each donor's sections
    subsets = ([adata[adata.obs["donor"] == d].copy() for d in sorted(adata.obs["donor"].unique())]
               if per_donor else [adata])
    for sub in subsets:
        sub.obs["batch"] = sub.obs["batch"].cat.remove_unused_categories()
    for seed in args.seeds:
        row = {"method": args.method, "config": args.setting, "seed": seed, "params": str(params),
               "space": "per_donor" if per_donor else "joint"}
        try:
            import torch
            row["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass
        try:
            with C.ResourceMeter() as meter:
                emb = pd.concat([METHODS[args.method](sub, seed, **params) for sub in subsets])
            Z = emb.reindex(adata.obs_names).to_numpy(np.float32)      # NaN rows = dropped spots
            np.save(out / f"{args.setting}_s{seed}.npy", Z)
            row.update(status="ok", seconds=meter.seconds, peak_gb=meter.peak_gb,
                       n_missing=int(np.isnan(Z).any(1).sum()), dim=Z.shape[1])
            try:
                import torch
                row["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            except ImportError:
                pass
        except Exception as e:      # failures are results: keep them
            row.update(status=f"failed: {type(e).__name__}: {e}"[:300])
            traceback.print_exc()
        C.append_rows([row], out / "runs.tsv")
        if row["status"].startswith(("failed: ImportError", "failed: ModuleNotFoundError", "failed: OSError")):
            break                   # environment problem: other seeds would fail the same way


if __name__ == "__main__":
    main()
