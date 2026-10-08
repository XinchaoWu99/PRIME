"""Assemble the Visium HD mouse brain cohort (16 um bins): bin QC, cell-type reference, dentate-gyrus tiles.

    python prepare_cohort.py [--tile-um 1980]

Three public 10x samples of the same tissue prepared three ways (FFPE, fresh frozen, fixed frozen;
config.yaml `visiumhd_brain.samples`), so batch = preservation protocol. Needs CellTypist.

QC, applied per sample and identically for every method:
  1. debris: drop connected pieces of < 50 bins (8-neighbour grid of tissue bins);
  2. tissue edge: drop bins with fewer than 5 of their 8 grid neighbours in tissue;
  3. depth: UMI >= max(100, median - 3 MAD of log1p UMI) and detected genes >= max(100,
     median - 3 MAD of log1p genes), MAD scaled to the normal SD;
  4. mitochondrial fraction <= max(0.25, median + 3 MAD). The floor of 0.25 keeps mitochondria-rich tissue
     (hippocampal neuropil layers, pial surface), which a purely MAD-based cut removes as whole regions;
  5. genes detected in >= 20 bins in every sample.

Cell-type reference (never seen by an integration method): every sample is annotated on its own with CellTypist
(model Mouse_Whole_Brain = Allen whole mouse brain atlas subclasses, Yao et al. 2023) on log1p CP10k counts with
majority voting over the sample's own over-clustering. Subclasses are mapped to the atlas classes with the
published taxonomy table; a bin keeps a label only when its own best match and the majority vote agree ("nan"
otherwise). The region / layer labels used for scoring are made afterwards by annotate_regions.py, which uses
this per-bin reference only as supporting evidence.

Tiles: one contiguous square per sample, centred on the dentate gyrus, found label-blind as the densest 200 um
grid cell of Prox1-positive bins. Side 1980 um keeps the three tiles together below 46,340 bins (= sqrt(INT_MAX)),
so that methods building a dense bin x bin matrix can run.

Sequence of the Visium HD analysis: prepare_cohort.py; annotate_regions.py domain / subcluster / apply /
cortex_tile / labelmap; ../spatial/run_methods.py with `--dataset brain_dg_tiles | brain_cortex_tiles |
brain_sections`; evaluate_integration.py and evaluate_per_region.py once per seed (`--tag s<seed>`);
embedding_umap.py for the cortex tiles.

Writes (paths of config.yaml `datasets`):
  brain_sections    all QC-passed bins, raw counts, reference labels in obs
  brain_dg_tiles    the three dentate-gyrus tiles
and next to them labels_<sample>.tsv (CellTypist output per bin) and qc_manifest.tsv (md5 of the input matrix,
bins before / after QC, thresholds, share removed by each rule, label retention, tile size).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import common as C  # noqa: E402

BIN = "square_016um"


def md5(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def load16(name: str, d: Path) -> ad.AnnData:
    b = d / "binned_outputs" / BIN
    a = sc.read_10x_h5(b / "filtered_feature_bc_matrix.h5")
    a.var_names_make_unique()
    pos = pd.read_parquet(b / "spatial" / "tissue_positions.parquet").set_index("barcode").loc[a.obs_names]
    um = json.loads((b / "spatial" / "scalefactors_json.json").read_text())["microns_per_pixel"]
    a.obsm["spatial"] = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(float) * um
    a.obs["array_row"] = pos["array_row"].to_numpy().astype(np.int64)
    a.obs["array_col"] = pos["array_col"].to_numpy().astype(np.int64)
    a.obs["sample"] = name
    a.obs["key"] = name + "_" + a.obs_names
    a.uns["md5"] = md5(b / "filtered_feature_bc_matrix.h5")
    return a


def grid_graph(r: np.ndarray, c: np.ndarray):
    """Component size and number of occupied 8-neighbours of every bin on the array grid."""
    key = {(x, y): i for i, (x, y) in enumerate(zip(r.tolist(), c.tolist()))}
    rows, cols = [], []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                for i, (x, y) in enumerate(zip(r.tolist(), c.tolist())):
                    j = key.get((x + dr, y + dc))
                    if j is not None:
                        rows.append(i)
                        cols.append(j)
    n = len(r)
    _, lab = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
    return np.bincount(lab)[lab], np.bincount(np.asarray(rows, dtype=np.int64), minlength=n)


def bin_qc(a: ad.AnnData) -> tuple[np.ndarray, dict]:
    cnt = np.asarray(a.X.sum(1)).ravel()
    gen = np.asarray((a.X > 0).sum(1)).ravel()
    mt = a.var_names.str.upper().str.startswith("MT-")
    mtf = np.asarray(a.X[:, mt].sum(1)).ravel() / np.maximum(cnt, 1)
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    lo = lambda x: np.expm1(np.median(np.log1p(x)) - 3 * mad(np.log1p(x)))
    cut = {"min_umi": max(100.0, lo(cnt)), "min_genes": max(100.0, lo(gen)),
           "max_mt": max(0.25, np.median(mtf) + 3 * mad(mtf))}
    comp, nbrs = grid_graph(a.obs["array_row"].to_numpy(), a.obs["array_col"].to_numpy())
    fail = {"debris": comp < 50, "edge": nbrs < 5, "umi": cnt < cut["min_umi"], "genes": gen < cut["min_genes"],
            "mt": mtf > cut["max_mt"]}
    keep = ~np.any(list(fail.values()), axis=0)
    a.obs["total_counts"], a.obs["n_genes"], a.obs["mt_frac"] = cnt, gen, mtf
    stats = {**cut, **{f"removed_{k}": float(v.mean()) for k, v in fail.items()}, "removed_any": float(1 - keep.mean())}
    return keep, stats


def annotate(a: ad.AnnData, model: str, taxonomy: pd.DataFrame) -> pd.DataFrame:
    """CellTypist subclass per bin (best match and majority vote) and its Allen class."""
    import celltypist
    x = a.copy()
    sc.pp.normalize_total(x, target_sum=1e4)
    sc.pp.log1p(x)
    pred = celltypist.annotate(x, model=model, majority_voting=True)
    df = pred.predicted_labels.astype(str)          # categorical columns with different categories
    df["conf_score"] = pred.probability_matrix.max(1)
    sub2class = taxonomy.drop_duplicates("subclass").set_index("subclass")["class"]
    df["class_best"] = df["predicted_labels"].map(sub2class)
    df["class_majority"] = df["majority_voting"].map(sub2class)
    agree = df["class_best"] == df["class_majority"]
    df["label_class"] = df["class_majority"].where(agree, "nan").astype(str)
    df["label_subclass"] = df["majority_voting"].where(df["predicted_labels"] == df["majority_voting"], "nan").astype(str)
    return df


def dg_tile(a: ad.AnnData, side_um: float, cell_um: float = 200) -> np.ndarray:
    """Square centred on the densest `cell_um` grid cell of Prox1-positive bins (label-blind)."""
    xy = a.obsm["spatial"]
    pos = np.asarray(a[:, "Prox1"].X.sum(1)).ravel() > 0
    g = np.floor((xy[pos] - xy.min(0)) / cell_um).astype(int)
    cells, n = np.unique(g, axis=0, return_counts=True)
    c = xy.min(0) + (cells[n.argmax()] + 0.5) * cell_um
    return (np.abs(xy - c) <= side_um / 2).all(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-um", type=float, default=1980)
    args = ap.parse_args()
    cfg = C.load_config()
    hd = cfg["visiumhd_brain"]
    sections_path = Path(cfg["datasets"]["brain_sections"]["path"])
    tiles_path = Path(cfg["datasets"]["brain_dg_tiles"]["path"])
    out = sections_path.parent
    out.mkdir(parents=True, exist_ok=True)
    samples = {k: Path(v) for k, v in hd["samples"].items()}
    taxonomy = pd.read_csv(hd["taxonomy"])

    parts, rows = [], []
    for name, d in samples.items():
        a = load16(name, d)
        keep, stats = bin_qc(a)
        row = {"sample": name, "matrix_md5": a.uns.pop("md5"), "n_bins_raw": a.n_obs, "n_bins_qc": int(keep.sum()), **stats}
        a = a[keep].copy()
        parts.append(a)
        rows.append(row)
        print(row, flush=True)

    # genes detected in >= 20 bins in every sample
    ok = np.all([np.asarray((p.X > 0).sum(0)).ravel() >= 20 for p in parts], axis=0)
    genes = parts[0].var_names[ok]
    tiles = []
    for a, row in zip(parts, rows):
        a._inplace_subset_var(genes)
        name = row["sample"]
        lab = annotate(a, hd["celltypist_model"], taxonomy)
        lab.to_csv(out / f"labels_{name}.tsv", sep="\t")
        a.obs = a.obs.join(lab[["majority_voting", "predicted_labels", "conf_score", "label_class", "label_subclass"]])
        label, t = a.obs["label_class"], dg_tile(a, args.tile_um)
        row.update(n_genes_kept=len(genes), median_umi_qc=float(np.median(a.obs["total_counts"])),
                   median_genes_qc=float(np.median(a.obs["n_genes"])), labelled=float((label.astype(str) != "nan").mean()),
                   n_labels=int(label[label.astype(str) != "nan"].nunique()), tile_bins=int(t.sum()))
        a.obs_names = a.obs["key"].astype(str)
        tiles.append(a[t].copy())
        print(name, {k: row[k] for k in ("n_genes_kept", "median_umi_qc", "labelled", "n_labels", "tile_bins")}, flush=True)

    pd.DataFrame(rows).to_csv(out / "qc_manifest.tsv", sep="\t", index=False)
    ad.concat(parts, join="inner", merge="same").write_h5ad(sections_path)
    tl = ad.concat(tiles, join="inner", merge="same")
    assert tl.n_obs < 46340, tl.n_obs
    tl.write_h5ad(tiles_path)
    print("tiles:", tl.n_obs, tl.obs.groupby("sample").size().to_dict())


if __name__ == "__main__":
    main()
