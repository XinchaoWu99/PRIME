"""Anchor validation: are the consensus anchors correct, and do the projections vote independently?

    python anchor_validation.py [DATASET] [--k-max 100]

DATASET defaults to the Jurkat / 293T mixture, whose two cell lines give an unambiguous ground truth. For each
seed, the MNN sets of K_max projections are computed once; every (K, tau) consensus is a slice of them.
Outputs (out_dir/anchor_validation/DATASET):
  anchor_metrics.tsv   precision (+95% CI), FDR, proxy TPR, FPR, coverage per seed x K x tau
  coverage.tsv         same-type anchor coverage per batch x cell type
  condorcet.tsv        observed vs independent-voter consensus rates, mean phi, per class
                       (all pairs and per batch pair)
  vote_hist.tsv        vote counts of candidate pairs found by >= 1 projection
  phi_K{K}_{cls}_s{seed}.tsv  projection x projection error correlation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import anchors as A
import common as C

KS = [1, 5, 10, 20, 50, 100]
TAUS = [0.3, 0.5, 0.7, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="jurkat")
    ap.add_argument("--k-max", type=int, default=max(KS))
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    ds = cfg["datasets"][args.dataset]
    p = ds["default"]
    ks = sorted({k for k in KS + [p["n_projections"]] if k <= args.k_max})
    taus = sorted(set(TAUS + [p["consensus_threshold"]]))
    out = Path(cfg["out_dir"]) / "anchor_validation" / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    adata = C.load_dataset(args.dataset, cfg)
    lab, bat = adata.obs["label"].values, adata.obs["batch"].values
    bpairs = A.searched_batch_pairs(np.asarray(bat).astype(str))

    metrics, cover, cond, hist = [], [], [], []
    for seed in ds.get("seeds", cfg["seeds"]):
        votes = A.projection_votes(adata.X, bat, args.k_max, p["target_dim"], p["k_neighbors"], seed)
        for K in ks:
            for tau in taus if K > 1 else [taus[0]]:
                i, j, _ = A.consensus(votes, K, tau)
                m, cov = A.anchor_metrics(i, j, lab, bat)
                y = np.asarray(lab)[i] == np.asarray(lab)[j]
                lo, hi = A.precision_ci(i, y, seed=seed) if len(i) else (np.nan, np.nan)
                key = {"seed": seed, "K": K, "tau": tau, "min_votes": A.min_votes(K, tau)}
                metrics.append({**key, **m, "precision_lo": lo, "precision_hi": hi})
                cover.append(cov.assign(**key))
            if K == 1:
                continue
            for pair in [None, *bpairs]:
                s, phis, h = A.vote_structure(votes, K, taus, lab, bat, only_pair=pair)
                tag = "all" if pair is None else f"{pair[0]}|{pair[1]}"
                cond.append(s.assign(seed=seed, batch_pair=tag))
                if pair is None:
                    hist.append(h.assign(seed=seed, K=K))
                    for cls, phi in phis.items():
                        pd.DataFrame(phi).to_csv(out / f"phi_K{K}_{cls}_s{seed}.tsv", sep="\t", index=False)

    pd.DataFrame(metrics).to_csv(out / "anchor_metrics.tsv", sep="\t", index=False)
    pd.concat(cover).to_csv(out / "coverage.tsv", sep="\t", index=False)
    pd.concat(cond).to_csv(out / "condorcet.tsv", sep="\t", index=False)
    pd.concat(hist).to_csv(out / "vote_hist.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
