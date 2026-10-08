"""Per-region / per-layer view of cross-section alignment, on the same embeddings and labels as evaluate_integration.py.

    python evaluate_per_region.py DATASET METHOD=emb.npy [METHOD=emb.npy ...] --labels-tsv LABELS --tag TAG

For every method and every region scored in all sections (>= 20 bins each): the fraction of a bin's 15 nearest
neighbours (embedding, evaluation view) that come from another section (cross_nn; fully mixed = 1 - sum of squared
section shares), and among those cross-section neighbours the fraction carrying the same label (cross_same_label =
whether the sections are aligned region-to-region, e.g. L5 onto L5). Embeddings wider than eval_pcs are reduced to
their top PCs over all bins (as in evaluate_integration.py), then the kNN is taken over the bins of the scored regions.
Writes out_dir/visiumhd/<DATASET>/eval/<tag>/per_label.tsv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import common as C
from evaluate_integration import knn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("embeddings", nargs="+")
    ap.add_argument("--labels-tsv", required=True)
    ap.add_argument("--label-col", default="label_region")
    ap.add_argument("--tag", default="region_s0")
    args = ap.parse_args()
    cfg = C.load_config()
    out = Path(cfg["out_dir"]) / "visiumhd" / args.dataset / "eval" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    adata = C.load_dataset(args.dataset, cfg)
    reg = pd.read_csv(args.labels_tsv, sep="\t", index_col=0)[args.label_col].reindex(adata.obs_names).fillna("nan").astype(str)
    sec = adata.obs["batch"].astype(str)
    n = pd.crosstab(reg[reg != "nan"], sec[reg != "nan"])
    ok = n.index[(n >= 20).all(1)]
    keep = reg.isin(ok).to_numpy()
    lab, pat = reg.to_numpy()[keep], sec.to_numpy()[keep]
    rows = []
    for m, p in (e.split("=", 1) for e in args.embeddings):
        Z = np.load(p)
        ok_rows = np.isfinite(Z).all(1)
        assert ok_rows.all(), f"{m}: {int((~ok_rows).sum())} bins not returned"
        Z = C.eval_view(Z, cfg["eval_pcs"])[keep]      # as evaluate_integration.py: top PCs of all bins, then the scored bins
        nb = knn(Z, 15)
        cross = pat[nb] != pat[:, None]
        same = lab[nb] == lab[:, None]
        for k in ok:
            i = lab == k
            c = cross[i]
            rows.append({"method": m, "label": k, "n_bins": int(i.sum()), "cross_nn": c.mean(),
                         "cross_nn_mixed_expectation": 1 - pd.Series(pat[i]).value_counts(normalize=True).pow(2).sum(),
                         "cross_same_label": (same[i] & c).sum() / max(c.sum(), 1), "n_cross_pairs": int(c.sum())})
        print(m, "done", flush=True)
    pd.DataFrame(rows).to_csv(out / "per_label.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
