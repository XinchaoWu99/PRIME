"""Cluster-first, marker-based region / cortical-layer annotation of the Visium HD mouse brain sections (16 um bins).

    python annotate_regions.py cluster    [--res 0.5 1 2] [--subsample N]     # expression only (cell-type level)
    python annotate_regions.py domain     [--res 0.3 0.6 1 1.5] [--k 24 --lam 0.8]
    python annotate_regions.py subcluster subcluster_spec.tsv                 # e.g. cortex -> layers
    python annotate_regions.py replot     [--genes]                           # redraw maps (orientation)
    python annotate_regions.py apply      region_annotation.tsv
    python annotate_regions.py cortex_tile                                    # cortex + hippocampus tiles
    python annotate_regions.py labelmap                                       # overview figure of the labels

Produces the labels on which the integration of the three sections is scored. At 16 um a bin mixes several
cells, so per-bin cell-type calls resolve neither cortical layers nor regions; the labels are therefore
anatomical: regions first, then layers / fields inside cortex and hippocampus, as in spatial-transcriptomics
practice. Every cluster is annotated from marker genes; the per-bin CellTypist calls of prepare_cohort.py
(Allen whole mouse brain subclasses) are only quoted as a reference. Each sample is processed on its own, and
no integration method or integration result is used.

cluster     expression only: log1p CP10k, 3,000 HVGs, PCA 50, 15-NN graph on 30 PCs, Leiden. Cell-type-level
            clusters (glia, vascular and neuropil clusters are scattered across regions).
domain      region-level clusters: the bin's own PCs concatenated with the Gaussian-weighted mean PCs of its k
            nearest bins (k = 24 = the 5 x 5 block, ~40 um), each block scaled to unit total variance and weighted
            sqrt(1 - lam) / sqrt(lam) with lam = 0.8 (the "domain" setting of BANKSY, Singhal et al. 2024,
            Nat Genet 56:431); 15-NN graph, Leiden.
subcluster  Leiden on the same features restricted to the bins of chosen domain clusters (SPEC: sample,
            resolution, clusters, name, sub_res), e.g. isocortex -> layers, hippocampus -> fields.
apply       ANNOTATION (sample, level = dom_r<res> | sub_<name>, cluster, annotation, note; "nan" = unlabelled):
            domain rows first, sub-cluster rows override them; writes annot/labels_region.tsv (label_region).
cortex_tile writes the data set `brain_cortex_tiles` (see cmd_cortex_tile).
For every clustering the script writes the material used for annotation: mean expression / detection of a
curated marker panel (MARKERS) and a per-region marker score, top Wilcoxon DE genes, CellTypist composition
(reference only), bin counts in / outside the dentate-gyrus tile, and spatial maps. All outputs go to the folder
`annot/` next to the cohort file (config.yaml `datasets.brain_sections.path`).

The curated annotation of the three sections is region_annotation.tsv (with the marker evidence of every cluster)
and the sub-clustering specification is subcluster_spec.tsv. Leiden cluster ids depend on the software versions
and on the thread count of the neighbour search; the two tables refer to the clustering obtained with
single-threaded neighbours (`sc.settings.n_jobs = 1`, scanpy 1.12) and must be re-curated if the ids change.

Region vocabulary and markers follow the Allen CCFv3 parcellation (Wang et al. 2020, Cell 181:936) and
published layer / field markers: neocortical layers (Belgard et al. 2011, Neuron 71:605; Tasic et al. 2018,
Nature 563:72), hippocampal fields (Cembrowski et al. 2016, eLife 5:e14997), whole-brain classes (Yao et al.
2023, Nature 624:317); spatial-transcriptomic region atlas as precedent (Ortiz et al. 2020, Sci Adv 6:eabb3446).
"""
from __future__ import annotations

import argparse
import re
import warnings
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import common as C

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

MARKERS = {   # region / layer -> markers (mouse symbols)
    "neuron": ["Snap25", "Rbfox3", "Slc17a7", "Slc17a6", "Gad1", "Gad2"],
    "CTX-L1": ["Reln", "Ndnf", "Cxcl14", "Lamp5"],
    "CTX-L2/3": ["Cux2", "Otof", "Calb1", "Stard8", "Rasgrf2"],
    "CTX-L4": ["Rorb", "Rspo1", "Scnn1a", "Whrn"],
    "CTX-L5": ["Fezf2", "Bcl11b", "Deptor", "Parm1", "Pou3f1", "Etv1", "Tshz2"],
    "CTX-L6": ["Foxp2", "Syt6", "Tle4", "Rprm", "Crym", "Sulf1"],
    "CTX-L6b": ["Ccn2", "Ctgf", "Cplx3", "Nxph4"],
    "CA1": ["Fibcd1", "Wfs1", "Mpped1"],
    "CA2": ["Amigo2", "Pcp4", "Map3k15"],
    "CA3": ["Bok", "Cpne4", "Nectin3"],
    "DG": ["Prox1", "C1ql2", "Dsp", "Pdzd2"],
    "TH": ["Tcf7l2", "Prkcd", "Synpo2", "Rora"],
    "RT": ["Pvalb", "Ecel1"],
    "MH/LH": ["Tac2", "Pou4f1", "Gpr151", "Chrna3"],
    "HY": ["Otp", "Sim1", "Gal", "Pmch", "Hcrt"],
    "STR": ["Ppp1r1b", "Penk", "Drd1", "Drd2", "Adora2a", "Rgs9"],
    "MB": ["Th", "Slc6a3"],
    "fiber tracts": ["Mbp", "Plp1", "Mobp", "Mog", "Mal"],
    "astro": ["Gfap", "Aqp4", "Slc1a2", "Aldh1l1"],
    "ventricle": ["Foxj1", "Ccdc153", "Rarres2", "Tmem212"],
    "choroid plexus": ["Ttr", "Kl", "Folr1"],
    "meninges": ["Ptgds", "Dcn", "Slc6a13", "Col1a2"],
    "vascular": ["Cldn5", "Flt1", "Vtn", "Acta2"],
    "microglia": ["Cx3cr1", "P2ry12", "Hexb"],
    "OPC": ["Pdgfra"],
}
MAP_GENES = ["Reln", "Cux2", "Rorb", "Fezf2", "Bcl11b", "Foxp2", "Syt6", "Ccn2", "Cplx3", "Fibcd1", "Amigo2",
             "Bok", "Prox1", "C1ql2", "Tcf7l2", "Prkcd", "Pvalb", "Tac2", "Ppp1r1b", "Mbp", "Gfap", "Ttr",
             "Foxj1", "Ptgds"]


def out_dir(cfg, subsample: bool = False) -> Path:
    d = Path(cfg["datasets"]["brain_sections"]["path"]).parent / ("annot_subsample" if subsample else "annot")
    d.mkdir(parents=True, exist_ok=True)
    return d


def view(xy: np.ndarray, sample: str, cfg=None):
    """Display coordinates: every section drawn like FFPE (dorsal up, midline right); config visiumhd_brain.display."""
    M = np.asarray((cfg or C.load_config())["visiumhd_brain"]["display"][sample], float)
    v = np.c_[xy[:, 0], -xy[:, 1]] @ M.T         # image rows grow downwards
    return v[:, 0], v[:, 1]


def preprocess(a: sc.AnnData, seed: int = 0) -> sc.AnnData:
    """log1p CP10k in X; PCA (50) of the scaled 3,000 HVGs in obsm['X_pca']."""
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=3000, flavor="seurat")
    x = a[:, a.var["highly_variable"]].copy()
    sc.pp.scale(x, max_value=10)
    sc.tl.pca(x, n_comps=50, random_state=seed)
    a.obsm["X_pca"] = x.obsm["X_pca"]
    return a


def leiden(a: sc.AnnData, res: list[float], prefix: str, seed: int = 0, **nn):
    sc.pp.neighbors(a, n_neighbors=15, random_state=seed, **nn)
    for r in res:
        sc.tl.leiden(a, resolution=r, key_added=f"{prefix}_r{r:g}", random_state=seed, flavor="igraph",
                     n_iterations=2, directed=False)


def cluster_sample(a: sc.AnnData, res: list[float], seed: int = 0) -> sc.AnnData:
    a = preprocess(a, seed)
    leiden(a, res, "leiden", seed, n_pcs=30, use_rep="X_pca")
    return a


def spatial_features(pc: np.ndarray, xy: np.ndarray, k: int = 24, lam: float = 0.8) -> np.ndarray:
    """[sqrt(1-lam) * own PCs, sqrt(lam) * Gaussian-weighted mean PCs of the k nearest bins], blocks at unit variance."""
    d, idx = cKDTree(xy).query(xy, k + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    sigma = np.median(d[:, -1]) / 2
    w = np.exp(-d ** 2 / (2 * sigma ** 2))
    w /= w.sum(1, keepdims=True)
    nbr = np.einsum("nk,nkd->nd", w, pc[idx])
    unit = lambda z: z / np.sqrt(z.var(0).sum())
    return np.hstack([np.sqrt(1 - lam) * unit(pc), np.sqrt(lam) * unit(nbr)]).astype(np.float32)


def natural(c: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", c)]


def summarise(a: sc.AnnData, key: str, ct: pd.DataFrame, tile: np.ndarray, out: Path, tag: str, sample: str):
    cl = a.obs[key].astype(str)
    order = sorted(cl.unique(), key=natural)
    genes = [g for gs in MARKERS.values() for g in gs if g in a.var_names]
    X = a[:, genes].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    mean = pd.DataFrame(X, columns=genes).groupby(cl.values).mean().loc[order]
    frac = pd.DataFrame(X > 0, columns=genes).groupby(cl.values).mean().loc[order]
    mean.to_csv(out / f"marker_mean_{tag}.tsv", sep="\t")
    frac.to_csv(out / f"marker_frac_{tag}.tsv", sep="\t")

    # region scores: mean over markers of the z-scored (across clusters) cluster means
    z = (mean - mean.mean()) / mean.std().replace(0, 1)
    score = pd.DataFrame({r: z[[g for g in gs if g in z]].mean(1) for r, gs in MARKERS.items()
                          if any(g in z for g in gs)})
    score.to_csv(out / f"region_score_{tag}.tsv", sep="\t")

    # DE on at most 600 bins per cluster
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.flatnonzero(cl.values == c), min(600, (cl.values == c).sum()), replace=False)
                          for c in order])
    s = a[np.sort(idx)].copy()
    s.obs[key] = s.obs[key].astype(str).astype("category")
    sc.tl.rank_genes_groups(s, key, method="wilcoxon", n_genes=25, use_raw=False)
    de = {c: [n for n in s.uns["rank_genes_groups"]["names"][c][:15]] for c in order}

    xy = a.obsm["spatial"]
    rows = []
    for c in order:
        m = (cl == c).to_numpy()
        comp = ct.loc[m, "predicted_labels"].value_counts(normalize=True).head(3)
        mv = ct.loc[m, "majority_voting"].value_counts(normalize=True).head(2)
        top = score.loc[c].sort_values(ascending=False).head(3)
        rows.append({"cluster": c, "n_bins": int(m.sum()), "n_in_tile": int((m & tile).sum()),
                     "frac_in_tile": float((m & tile).sum() / max(m.sum(), 1)),
                     "x_um": float(np.median(xy[m, 0])), "y_um": float(np.median(xy[m, 1])),
                     "median_umi": float(np.median(a.obs["total_counts"].to_numpy()[m])),
                     "top_region_scores": "; ".join(f"{r} {v:.2f}" for r, v in top.items()),
                     "celltypist_best": "; ".join(f"{k} {v:.2f}" for k, v in comp.items()),
                     "celltypist_majority": "; ".join(f"{k} {v:.2f}" for k, v in mv.items()),
                     "top_de": ", ".join(de[c])})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / f"cluster_summary_{tag}.tsv", sep="\t", index=False)
    plot_clusters(cl.to_numpy(), xy, tile, summ, score, out, tag, sample)


def plot_clusters(cl: np.ndarray, xy: np.ndarray, tile: np.ndarray, summ: pd.DataFrame, score: pd.DataFrame,
                  out: Path, tag: str, sample: str):
    """Whole map, one panel per cluster, heat map of the region marker scores."""
    cl = pd.Series(np.asarray(cl).astype(str))
    order = [str(c) for c in summ["cluster"]]
    summ = summ.assign(cluster=summ["cluster"].astype(str))
    score = score.set_axis(score.index.astype(str))
    x, y = view(xy, sample)
    cmap = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors + plt.get_cmap("tab20c").colors
    fig, ax = plt.subplots(figsize=(12, 9))
    for i, c in enumerate(order):
        m = (cl == c).to_numpy()
        ax.scatter(x[m], y[m], s=0.3, color=cmap[i % len(cmap)], rasterized=True)
        ax.text(np.median(x[m]), np.median(y[m]), c, fontsize=9, weight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, lw=0))
    if tile.any():
        tx, ty = x[tile], y[tile]
        ax.plot([tx.min(), tx.max(), tx.max(), tx.min(), tx.min()], [ty.min(), ty.min(), ty.max(), ty.max(), ty.min()], "k--", lw=1)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(f"{tag}: Leiden clusters (dashed = scored tile; dorsal up, midline right)")
    fig.savefig(out / f"map_{tag}.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    n = len(order); nc = 6; nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(nc * 3.2, nr * 2.6))
    for ax, c in zip(axs.ravel(), order):
        m = (cl == c).to_numpy()
        ax.scatter(x[~m], y[~m], s=0.05, color="0.88", rasterized=True)
        ax.scatter(x[m], y[m], s=0.15, color="crimson", rasterized=True)
        r = summ.set_index("cluster").loc[c]
        ax.set_title(f"{c} (n={r.n_bins}) {r.top_region_scores.split(';')[0]}", fontsize=7)
        ax.set_aspect("equal"); ax.axis("off")
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out / f"panels_{tag}.png", dpi=90); plt.close(fig)

    fig, ax = plt.subplots(figsize=(0.45 * score.shape[1] + 2, 0.3 * n + 2))
    im = ax.imshow(score.loc[order].to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(score.shape[1])); ax.set_xticklabels(score.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels([f"{c} ({summ.n_bins.iloc[i]})" for i, c in enumerate(order)], fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.5, label="mean z of marker cluster means")
    ax.set_title(f"{tag}: region marker scores")
    fig.tight_layout(); fig.savefig(out / f"scores_{tag}.png", dpi=100); plt.close(fig)


def gene_maps(a: sc.AnnData, out: Path, tag: str, sample: str):
    genes = [g for g in MAP_GENES if g in a.var_names]
    x, y = view(a.obsm["spatial"], sample)
    nc = 6; nr = int(np.ceil(len(genes) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(nc * 3.2, nr * 2.6))
    for ax, g in zip(axs.ravel(), genes):
        v = np.asarray(a[:, g].X.todense()).ravel()
        o = np.argsort(v)
        ax.scatter(x[o], y[o], c=v[o], s=0.1, cmap="magma_r", vmax=np.quantile(v[v > 0], 0.99) if (v > 0).any() else 1,
                   rasterized=True)
        ax.set_title(g, fontsize=8); ax.set_aspect("equal"); ax.axis("off")
    for ax in axs.ravel()[len(genes):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out / f"genes_{tag}.png", dpi=90); plt.close(fig)


def load_inputs(cfg):
    full = sc.read_h5ad(cfg["datasets"]["brain_sections"]["path"])
    tiles = sc.read_h5ad(cfg["datasets"]["brain_dg_tiles"]["path"], backed="r")
    return full, set(tiles.obs_names)


def celltypist_ref(cfg, name: str, index) -> pd.DataFrame:
    lab_dir = Path(cfg["datasets"]["brain_sections"]["path"]).parent
    ct = pd.read_csv(lab_dir / f"labels_{name}.tsv", sep="\t", index_col=0)
    ct.index = name + "_" + ct.index.astype(str)
    return ct.reindex(index)


def cmd_cluster(args, cfg):
    out = out_dir(cfg, subsample=bool(args.subsample))
    full, tile_keys = load_inputs(cfg)
    missing = [g for gs in MARKERS.values() for g in gs if g not in full.var_names]
    print("markers not in the panel:", missing, flush=True)
    allc = []
    for name in full.obs["sample"].unique():
        a = full[full.obs["sample"] == name].copy()
        if args.subsample:
            a = a[np.random.default_rng(0).choice(a.n_obs, args.subsample, replace=False)].copy()
        a = cluster_sample(a, args.res)
        ct = celltypist_ref(cfg, name, a.obs_names)
        tile = a.obs_names.isin(list(tile_keys))
        gene_maps(a, out, name, name)
        for r in args.res:
            summarise(a, f"leiden_r{r:g}", ct, tile, out, f"{name}_r{r:g}", name)
        c = a.obs[[f"leiden_r{r:g}" for r in args.res]].astype(str).copy()
        c.insert(0, "sample", name)
        c["x_um"], c["y_um"], c["in_tile"] = a.obsm["spatial"][:, 0], a.obsm["spatial"][:, 1], tile
        allc.append(c)
        print(name, {r: a.obs[f"leiden_r{r:g}"].nunique() for r in args.res}, flush=True)
    pd.concat(allc).rename_axis("key").to_csv(out / "clusters.tsv", sep="\t")


def cmd_domain(args, cfg):
    out = out_dir(cfg)
    full, tile_keys = load_inputs(cfg)
    allc = []
    for name in full.obs["sample"].unique():
        a = preprocess(full[full.obs["sample"] == name].copy())
        a.obsm["X_dom"] = spatial_features(a.obsm["X_pca"], a.obsm["spatial"], args.k, args.lam)
        np.save(out / f"domfeat_{name}.npy", a.obsm["X_dom"])
        leiden(a, args.res, "dom", use_rep="X_dom")
        ct = celltypist_ref(cfg, name, a.obs_names)
        tile = a.obs_names.isin(list(tile_keys))
        for r in args.res:
            summarise(a, f"dom_r{r:g}", ct, tile, out, f"dom_{name}_r{r:g}", name)
        c = a.obs[[f"dom_r{r:g}" for r in args.res]].astype(str).copy()
        c.insert(0, "sample", name)
        c["x_um"], c["y_um"], c["in_tile"] = a.obsm["spatial"][:, 0], a.obsm["spatial"][:, 1], tile
        allc.append(c)
        print(name, {r: a.obs[f"dom_r{r:g}"].nunique() for r in args.res}, flush=True)
    pd.concat(allc).rename_axis("key").to_csv(out / "domains.tsv", sep="\t")


def cmd_subcluster(args, cfg):
    out = out_dir(cfg)
    spec = pd.read_csv(args.spec, sep="\t", dtype=str, comment="#")
    dom = pd.read_csv(out / "domains.tsv", sep="\t", index_col=0, dtype=str)
    full, tile_keys = load_inputs(cfg)
    for name, rows in spec.groupby("sample"):
        a = full[full.obs["sample"] == name].copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        feat = np.load(out / f"domfeat_{name}.npy")
        d = dom.loc[a.obs_names]
        ct = celltypist_ref(cfg, name, a.obs_names)
        tile = a.obs_names.isin(list(tile_keys))
        for _, r in rows.iterrows():
            m = d[f"dom_r{r.resolution}"].isin(r.clusters.split(",")).to_numpy()
            b = a[m].copy()
            b.obsm["X_dom"] = feat[m]
            leiden(b, [float(r.sub_res)], "sub", use_rep="X_dom")
            key = f"sub_r{float(r.sub_res):g}"
            summarise(b, key, ct[m], tile[m], out, f"sub_{r['name']}_{name}", name)
            b.obs[[key]].astype(str).rename(columns={key: "cluster"}).rename_axis("key").to_csv(
                out / f"sub_{r['name']}_{name}.tsv", sep="\t")
            print(name, r["name"], int(m.sum()), "bins ->", b.obs[key].nunique(), "sub-clusters", flush=True)


def cmd_replot(args, cfg):
    """Redraw all maps from the saved assignments and summaries (no re-clustering), plus the gene maps."""
    out = out_dir(cfg)
    for fname, prefix, tagf in (("clusters.tsv", "leiden_r", "{s}_r{r}"), ("domains.tsv", "dom_r", "dom_{s}_r{r}")):
        if not (out / fname).exists():
            continue
        tab = pd.read_csv(out / fname, sep="\t", index_col=0, dtype=str)
        for s_, t in tab.groupby("sample", sort=False):
            xy = t[["x_um", "y_um"]].astype(float).to_numpy()
            tile = (t["in_tile"] == "True").to_numpy()
            for col in [c for c in t.columns if c.startswith(prefix)]:
                tag = tagf.format(s=s_, r=col[len(prefix):])
                summ = pd.read_csv(out / f"cluster_summary_{tag}.tsv", sep="\t")
                score = pd.read_csv(out / f"region_score_{tag}.tsv", sep="\t", index_col=0)
                plot_clusters(t[col].to_numpy(), xy, tile, summ, score, out, tag, s_)
    dom = pd.read_csv(out / "domains.tsv", sep="\t", index_col=0, dtype=str)
    for f in sorted(out.glob("sub_*_*.tsv")):
        tag = f.stem
        sub = pd.read_csv(f, sep="\t", index_col=0, dtype=str)["cluster"]
        s_ = dom.loc[sub.index[0], "sample"]
        t = dom.loc[sub.index]
        plot_clusters(sub.to_numpy(), t[["x_um", "y_um"]].astype(float).to_numpy(), (t["in_tile"] == "True").to_numpy(),
                      pd.read_csv(out / f"cluster_summary_{tag}.tsv", sep="\t"),
                      pd.read_csv(out / f"region_score_{tag}.tsv", sep="\t", index_col=0), out, tag, s_)
    if args.genes:
        full, _ = load_inputs(cfg)
        for s_ in full.obs["sample"].unique():
            a = full[full.obs["sample"] == s_].copy()
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
            gene_maps(a, out, s_, s_)


def ctx_tile_mask(xy: np.ndarray, sample: str, cfg, side: float = 1980.0, ml: float = 2000.0):
    """Cortex-tile rule for one section (see cmd_cortex_tile): mask of the bins in the tile, centre x and top edge (view)."""
    xv, yv = view(xy, sample, cfg)
    cx = np.quantile(xv, 0.99) - ml
    top = yv[np.abs(xv - cx) <= 100].max()
    return (np.abs(xv - cx) <= side / 2) & (yv <= top) & (yv >= top - side), cx, top


def cmd_cortex_tile(args, cfg):
    """Cortex tiles (data set `brain_cortex_tiles`), placed by a label-blind geometric rule: the dentate-gyrus tiles
    hold almost no isocortex, so cortical layers cannot be scored there. Same side (1980 um). In the standard view
    (dorsal up, midline right; config visiumhd_brain.display) the square is centred 2.0 mm lateral to the midline
    (midline = 99th percentile of x) and its top edge is the dorsal tissue edge at that position (highest bin within
    100 um), i.e. a somatosensory-cortex column: pia -> L1-L6b -> corpus callosum -> dorsal hippocampus."""
    full, _ = load_inputs(cfg)
    keep = np.zeros(full.n_obs, bool)
    for name in full.obs["sample"].unique():
        m = (full.obs["sample"] == name).to_numpy()
        t, cx, top = ctx_tile_mask(full.obsm["spatial"][m], name, cfg, args.side, args.ml)
        keep[np.flatnonzero(m)[t]] = True
        print(name, "centre x (view)", round(float(cx)), "top", round(float(top)), "bins", int(t.sum()), flush=True)
    assert keep.sum() < 46340, keep.sum()                           # sqrt(INT_MAX): dense bin x bin matrices
    path = Path(cfg["datasets"]["brain_cortex_tiles"]["path"])
    full[keep].copy().write_h5ad(path)
    lab = pd.read_csv(out_dir(cfg) / "labels_region.tsv", sep="\t", index_col=0)["label_region"].reindex(full.obs_names[keep])
    print("tiles:", int(keep.sum()), "->", path, flush=True)
    print(pd.crosstab(lab.fillna("nan"), full.obs["sample"].to_numpy()[keep]).to_string(), flush=True)


LABEL_ORDER = ["meninges", "CTX L1", "CTX L2/3", "CTX L4", "CTX L5", "CTX L6", "CTX L6b", "RSP", "CTX lateral", "CLA/EP",
               "PIR", "AMY BLA", "AMY MEA/COA", "sAMY/BST", "CA1", "CA3", "DG", "HIP neuropil", "fiber tracts",
               "STR", "STRv", "PAL", "TH Prkcd+", "TH Prkcd-", "RT", "HAB", "STN", "HY", "choroid plexus", "ependyma",
               "vessels"]
DOT_GENES = ["Ptgds", "Cxcl14", "Lamp5", "Calb1", "Rorb", "Whrn", "Etv1", "Fezf2", "Rprm", "Tle4", "Ccn2", "Cplx3",
             "Tshz2", "Nr4a2", "Fibcd1", "Bok", "Prox1", "C1ql2", "Ddn", "Mbp", "Ppp1r1b", "Penk", "Prkcd", "Tcf7l2",
             "Pvalb", "Tac2", "Hap1", "Ttr", "Foxj1", "Acta2"]


def cmd_labelmap(args, cfg):
    """Final region / layer labels of the three sections side by side (standard view) with both tile outlines, and a
    dot plot of marker means per label and section (the review figure for the annotation)."""
    out = out_dir(cfg)
    lab = pd.read_csv(out / "labels_region.tsv", sep="\t", index_col=0)
    dom = pd.read_csv(out / "domains.tsv", sep="\t", index_col=0)
    ctx_path = Path(cfg["datasets"]["brain_cortex_tiles"]["path"])
    ctx = set(sc.read_h5ad(ctx_path, backed="r").obs_names) if ctx_path.exists() else set()
    labs = [l for l in LABEL_ORDER if l in set(lab["label_region"])]
    pal = dict(zip(labs, (plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors)[:len(labs)]))
    samples = list(dict.fromkeys(dom["sample"]))
    fig, axs = plt.subplots(1, len(samples), figsize=(7 * len(samples), 7.5))
    for ax, s_ in zip(axs, samples):
        d = dom[dom["sample"] == s_]
        x, y = view(d[["x_um", "y_um"]].to_numpy(float), s_, cfg)
        l = lab.loc[d.index, "label_region"].astype(str).to_numpy()
        ax.scatter(x[l == "nan"], y[l == "nan"], s=0.2, color="0.85", rasterized=True)
        for k in labs:
            ax.scatter(x[l == k], y[l == k], s=0.2, color=pal[k], rasterized=True)
        for mask, ls in ((d["in_tile"].to_numpy(bool), "--"), (d.index.isin(list(ctx)), "-")):
            if mask.any():
                tx, ty = x[mask], y[mask]
                ax.plot([tx.min(), tx.max(), tx.max(), tx.min(), tx.min()], [ty.min(), ty.min(), ty.max(), ty.max(), ty.min()], "k", ls=ls, lw=1)
        ax.set_title(f"{s_} (dorsal up, midline right)"); ax.set_aspect("equal"); ax.axis("off")
    handles = [plt.Line2D([], [], ls="", marker="o", color=pal[k], label=k) for k in labs]
    handles += [plt.Line2D([], [], color="k", ls="--", label="DG-centred tile"), plt.Line2D([], [], color="k", label="cortex tile")]
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(out / "labels_region_map.png", dpi=130); plt.close(fig)

    full, _ = load_inputs(cfg)
    genes = [g for g in DOT_GENES if g in full.var_names]
    rows = []
    for s_ in samples:
        a = full[full.obs["sample"] == s_][:, genes].copy()
        a.X = a.X.multiply(1e4 / np.asarray(full[full.obs["sample"] == s_].X.sum(1))).tocsr()   # CP10k on all genes
        a.X.data = np.log1p(a.X.data)
        X = a.X.toarray()
        l = lab.loc[a.obs_names, "label_region"].astype(str).to_numpy()
        for k in labs:
            m = l == k
            if m.sum() >= 20:
                rows.append({"sample": s_, "label": k, "n": int(m.sum()),
                             **{f"mean:{g}": X[m, i].mean() for i, g in enumerate(genes)},
                             **{f"frac:{g}": (X[m, i] > 0).mean() for i, g in enumerate(genes)}})
    dot = pd.DataFrame(rows)
    dot.to_csv(out / "labels_region_markers.tsv", sep="\t", index=False)
    ylab = [f"{r.label} | {r.sample[:5]}" for r in dot.itertuples()]
    mean = dot[[f"mean:{g}" for g in genes]].to_numpy()
    frac = dot[[f"frac:{g}" for g in genes]].to_numpy()
    z = np.vstack([mean[i] / np.maximum(mean[(dot["sample"] == s_).to_numpy()].max(0), 1e-9)
                   for i, s_ in enumerate(dot["sample"])])                 # scaled to the gene max within the section
    fig, ax = plt.subplots(figsize=(0.38 * len(genes) + 3, 0.19 * len(dot) + 2))
    yy, xx = np.meshgrid(np.arange(len(dot)), np.arange(len(genes)), indexing="ij")
    ax.scatter(xx.ravel(), yy.ravel(), s=frac.ravel() * 40, c=z.ravel(), cmap="Reds", vmin=0, vmax=1, edgecolors="none")
    ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(dot))); ax.set_yticklabels(ylab, fontsize=6); ax.invert_yaxis()
    ax.set_xlim(-0.5, len(genes) - 0.5); ax.grid(False)
    ax.set_title("marker mean (colour, scaled to the gene max within the section) and detection rate (size)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "labels_region_dotplot.png", dpi=130); plt.close(fig)


def read_annotation(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, comment="#", keep_default_na=False)   # "nan" = leave unlabelled


def apply_annotation(dom: pd.DataFrame, subs: dict, ann: pd.DataFrame) -> pd.Series:
    """Region / layer label of every bin. dom: index = bin key, columns sample and dom_r<res> (domain clusters);
    subs: {(level, sample): Series bin key -> sub-cluster}; ann: the curated table. Domain levels are applied coarse
    to fine, then sub-cluster levels override the bins they cover (every sub-cluster must be listed)."""
    lab = pd.Series("nan", index=dom.index, name="label_region")
    order = lambda lv: (lv.startswith("sub_"), float(lv[5:]) if lv.startswith("dom_r") else 0.0)
    for level in sorted(ann["level"].unique(), key=order):
        for name, g in ann[ann["level"] == level].groupby("sample"):
            col = dom.loc[dom["sample"] == name, level] if level.startswith("dom_r") else subs[(level, name)]
            g = g.set_index("cluster")["annotation"]
            unknown = set(col.unique()) - set(g.index)
            assert not unknown or not level.startswith("sub_"), (level, name, unknown)
            col = col[col.isin(g.index)]
            lab.loc[col.index] = col.map(g).values
    return lab


def cmd_apply(args, cfg):
    out = out_dir(cfg)
    dom = pd.read_csv(out / "domains.tsv", sep="\t", index_col=0, dtype=str)
    ann = read_annotation(args.annotation)
    subs = {(lv, s_): pd.read_csv(out / f"{lv}_{s_}.tsv", sep="\t", index_col=0, dtype=str)["cluster"]
            for lv, s_ in ann.loc[ann["level"].str.startswith("sub_"), ["level", "sample"]].drop_duplicates().itertuples(index=False)}
    res = apply_annotation(dom, subs, ann).to_frame().join(dom[["sample", "in_tile"]])
    res.to_csv(out / "labels_region.tsv", sep="\t")
    print(pd.crosstab(res["label_region"], [res["sample"], res["in_tile"]]).to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cluster")
    p.add_argument("--res", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--subsample", type=int, default=0)
    p = sub.add_parser("domain")
    p.add_argument("--res", type=float, nargs="+", default=[0.3, 0.6, 1.0, 1.5])
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--lam", type=float, default=0.8)
    p = sub.add_parser("subcluster")
    p.add_argument("spec")
    p = sub.add_parser("replot")
    p.add_argument("--genes", action="store_true")
    p = sub.add_parser("labelmap")
    p = sub.add_parser("cortex_tile")
    p.add_argument("--side", type=float, default=1980.0)
    p.add_argument("--ml", type=float, default=2000.0, help="um lateral to the midline")
    p = sub.add_parser("apply")
    p.add_argument("annotation")
    args = ap.parse_args()
    cfg = C.load_config()
    {"cluster": cmd_cluster, "domain": cmd_domain, "subcluster": cmd_subcluster, "replot": cmd_replot,
     "cortex_tile": cmd_cortex_tile, "labelmap": cmd_labelmap, "apply": cmd_apply}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
