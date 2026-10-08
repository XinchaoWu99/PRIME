"""UMAP of the integrated embeddings of one Visium HD data set (display only: no metric is computed here).

    python embedding_umap.py DATASET METHOD=emb.npy [METHOD=emb.npy ...] --tag TAG --labels-tsv labels_region.tsv

Each embedding is reduced exactly as in evaluate_integration.py (top eval_pcs PCs of the bins that all given
methods returned); UMAP on a 15-NN graph of that view (scanpy defaults, random_state 0).
Output: out_dir/visiumhd/<DATASET>/umap/<tag>/umap.tsv.gz  (key, sample, label_region, <method>_1, <method>_2 per method)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("embeddings", nargs="+", help="METHOD=path.npy")
    ap.add_argument("--tag", default="s0")
    ap.add_argument("--labels-tsv", required=True)
    args = ap.parse_args()
    cfg = C.load_config()
    d = cfg["datasets"][args.dataset]
    obs = sc.read_h5ad(d["path"], backed="r").obs
    embs = {m: np.load(p) for m, p in (e.split("=", 1) for e in args.embeddings)}
    keep = np.all([np.isfinite(Z).all(1) for Z in embs.values()], axis=0)
    print(f"{keep.sum()} / {len(keep)} bins returned by all methods", flush=True)
    lab = pd.read_csv(args.labels_tsv, sep="\t", index_col=0)["label_region"].reindex(obs.index[keep])
    out = pd.DataFrame({"sample": obs[d["batch"]].astype(str).to_numpy()[keep],
                        "label_region": lab.fillna("nan").astype(str).to_numpy()}, index=obs.index[keep])
    for m, Z in embs.items():
        ad = sc.AnnData(obs=pd.DataFrame(index=out.index), obsm={"Z": C.eval_view(Z[keep], cfg["eval_pcs"])})
        sc.pp.neighbors(ad, use_rep="Z", n_neighbors=15, random_state=0)
        sc.tl.umap(ad, random_state=0)
        out[f"{m}_1"], out[f"{m}_2"] = ad.obsm["X_umap"][:, 0], ad.obsm["X_umap"][:, 1]
        print("umap", m, flush=True)
    dst = Path(cfg["out_dir"]) / "visiumhd" / args.dataset / "umap" / args.tag
    dst.mkdir(parents=True, exist_ok=True)
    out.rename_axis("key").to_csv(dst / "umap.tsv.gz", sep="\t")
    print("wrote", dst / "umap.tsv.gz", flush=True)


if __name__ == "__main__":
    main()
