"""Downstream validation of integrated DLPFC embeddings: spatial domains, boundaries, markers, label transfer.

    python domain_validation.py METHOD=emb.npy [METHOD=emb.npy ...] [--seeds 0 1 2 3 4] [--tag TAG]

Each .npy has one row per spot of DLPFC_merged.h5ad in file order (as written by run_methods.py); spots a
method did not return are NaN rows and are not imputed. Every method is scored on the spots that all methods
returned, with the same clustering and the same physical-space graph. Labels are used only for scoring.
Outputs (out_dir/domains/<tag>):
  domains_<method>_s<seed>.tsv   GMM (K = 7 layers, stated prior) and Leiden domains
  per_slice.tsv                  ARI/NMI, abnormal-spot %, CHAOS, distant-layer error,
                                 physical-neighbour retention, marker effect size
  donor_transfer.tsv             layer prediction for a held-out donor via kNN in the embedding
  slice_transfer.tsv             layer prediction for a held-out section from the other sections of its donor
                                 (cross-section alignment; defined for per-donor methods too)
  donor_marker_consistency.tsv   correlation of domain x marker-set profiles between donors
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import common as C
from prime.metrics.xlc import LAYER_TO_RANK  # noqa: E402


def knn_idx(X, k):
    return NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]


def abnormal_spots(xy, dom, k=10, min_diff=6):
    """% of spots whose domain differs from >= 6 of their 10 physical neighbours."""
    nb = knn_idx(xy, k)
    return float(((dom[nb] != dom[:, None]).sum(1) >= min_diff).mean())


def chaos(xy, dom):
    """Mean distance to the nearest same-domain spot (coordinates scaled to unit range)."""
    xy = (xy - xy.min(0)) / np.ptp(xy, 0).max()
    d = [NearestNeighbors(n_neighbors=2).fit(xy[dom == c]).kneighbors(xy[dom == c])[0][:, 1]
         for c in np.unique(dom) if (dom == c).sum() > 1]
    return float(np.concatenate(d).mean())


def neighbour_retention(xy, Z, k=10):
    """Fraction of each spot's k physical neighbours that are also embedding neighbours (same slice)."""
    a, b = knn_idx(xy, k), knn_idx(Z, k)
    return float(np.mean([len(np.intersect1d(x, y)) / k for x, y in zip(a, b)]))


def marker_scores(adata, markers: pd.DataFrame) -> pd.DataFrame:
    """Per-spot mean z-score (within slice) of each layer's marker set, from log-normalised counts."""
    ad = adata[:, adata.var_names.isin(markers["gene"])].copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    X = pd.DataFrame(ad.X.toarray(), index=ad.obs_names, columns=ad.var_names)
    X = X.groupby(ad.obs["batch"].values, observed=True).transform(lambda c: (c - c.mean()) / (c.std() + 1e-8))
    return pd.DataFrame({L: X[g["gene"][g["gene"].isin(X.columns)]].mean(1)
                         for L, g in markers.groupby("layer")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("embeddings", nargs="+", help="METHOD=path.npy")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--per-donor", nargs="*", default=[],
                    help="methods trained separately per donor (e.g. spamask): PCA and clustering are done "
                         "within each donor; cross-donor transfer/consistency are not defined for them")
    ap.add_argument("--tag", default="", help="sub-directory of out_dir/domains (e.g. one per embedding seed)")
    args = ap.parse_args()

    cfg = C.load_config()
    dcfg = cfg["domains"]
    out = Path(cfg["out_dir"]) / "domains" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    adata = C.load_dataset("dlpfc", cfg)
    embs = {m: np.load(p) for m, p in (e.split("=", 1) for e in args.embeddings)}
    common_spots = np.all([np.isfinite(Z).all(1) for Z in embs.values()], axis=0)
    adata, embs = adata[common_spots].copy(), {m: Z[common_spots] for m, Z in embs.items()}
    print(f"{common_spots.sum()} / {len(common_spots)} spots returned by all methods")

    lab = adata.obs["label"].astype(str).values
    rank = np.array([LAYER_TO_RANK.get(x, np.nan) for x in lab])
    labelled = np.isfinite(rank)
    sl, donor = adata.obs["batch"].astype(str).values, adata.obs["donor"].astype(str).values
    xy = np.asarray(adata.obsm["spatial"], float)
    markers = pd.read_csv(Path(__file__).resolve().parent / dcfg["markers"], sep="\t")
    ms = marker_scores(adata, markers)

    per_slice, transfer, consistency, slice_transfer = [], [], [], []
    for m, Z in embs.items():
        separate = m in args.per_donor
        groups = [donor == d for d in np.unique(donor)] if separate else [np.ones(len(donor), bool)]
        Zp = np.zeros((len(donor), min(20, Z.shape[1])), np.float32)
        for g in groups:
            Zp[g] = C.eval_view(Z[g], 20)
        for seed in args.seeds:
            doms = {"gmm": np.zeros(len(donor), int), "leiden": np.zeros(len(donor), int)}
            for n, g in enumerate(groups):          # domain ids are made unique across groups
                doms["gmm"][g] = 100 * n + GaussianMixture(dcfg["n_domains"], covariance_type="full", reg_covar=1e-4,
                                                           random_state=seed).fit_predict(Zp[g])
                ad = sc.AnnData(obs=adata.obs[[]].iloc[np.flatnonzero(g)].copy(), obsm={"Z": Zp[g]})
                sc.pp.neighbors(ad, use_rep="Z", n_neighbors=15, random_state=seed)
                sc.tl.leiden(ad, resolution=dcfg["leiden_resolution"], random_state=seed, flavor="igraph", n_iterations=2)
                doms["leiden"][g] = 100 * n + ad.obs["leiden"].astype(int).values
            pd.DataFrame(doms, index=adata.obs_names).to_csv(out / f"domains_{m}_s{seed}.tsv", sep="\t")

            for algo, dom in doms.items():
                # evaluation-only mapping of each domain to its majority layer
                dom_layer = pd.Series(rank[labelled]).groupby(dom[labelled]).agg(lambda r: r.mode().iloc[0])
                mapped = pd.Series(dom).map(dom_layer).values
                for s in np.unique(sl):
                    k = sl == s
                    kl = k & labelled
                    eff = []
                    for L, r in LAYER_TO_RANK.items():
                        if L not in ms or not (mapped[k] == r).any() or (mapped[k] == r).all():
                            continue
                        a, b = ms[L].values[k & (mapped == r)], ms[L].values[k & (mapped != r)]
                        eff.append((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12))
                    per_slice.append({
                        "method": m, "seed": seed, "clustering": algo, "slice": s, "donor": donor[k][0],
                        "n_domains": len(np.unique(dom)),
                        "ARI": adjusted_rand_score(lab[kl], dom[kl]), "NMI": normalized_mutual_info_score(lab[kl], dom[kl]),
                        "abnormal_spots": abnormal_spots(xy[k], dom[k]), "CHAOS": chaos(xy[k], dom[k]),
                        "distant_layer_error": float(np.mean(np.abs(mapped[kl] - rank[kl]) > 1)),
                        "marker_effect_size": np.mean(eff) if eff else np.nan,
                    })

                if separate:
                    continue
                # same domain -> same marker programme in every donor?
                prof = {d: ms[donor == d].groupby(dom[donor == d]).mean() for d in np.unique(donor)}
                for d1, d2 in combinations(prof, 2):
                    shared = prof[d1].index.intersection(prof[d2].index)
                    r = np.corrcoef(prof[d1].loc[shared].values.ravel(), prof[d2].loc[shared].values.ravel())[0, 1]
                    consistency.append({"method": m, "seed": seed, "clustering": algo, "donors": f"{d1}|{d2}", "r": r})

        for s in np.unique(sl):
            k = sl == s
            per_slice.append({"method": m, "seed": -1, "clustering": "none", "slice": s, "donor": donor[k][0],
                              "neighbour_retention": neighbour_retention(xy[k], Zp[k])})
        for d in np.unique(donor):   # within-donor alignment: fit on three slices, predict the fourth
            kd = labelled & (donor == d)
            for sh in np.unique(sl[kd]):
                tr, te = kd & (sl != sh), kd & (sl == sh)
                pred = KNeighborsClassifier(15).fit(Zp[tr], lab[tr]).predict(Zp[te])
                slice_transfer.append({"method": m, "donor": d, "held_out_slice": sh,
                                       "accuracy": float((pred == lab[te]).mean()),
                                       "macro_f1": f1_score(lab[te], pred, average="macro")})
        for d in ([] if separate else np.unique(donor)):    # label transfer: fit on two donors, predict the third
            tr, te = labelled & (donor != d), labelled & (donor == d)
            pred = KNeighborsClassifier(15).fit(Zp[tr], lab[tr]).predict(Zp[te])
            transfer.append({"method": m, "held_out_donor": d, "accuracy": float((pred == lab[te]).mean()),
                             "macro_f1": f1_score(lab[te], pred, average="macro")})

    pd.DataFrame(per_slice).to_csv(out / "per_slice.tsv", sep="\t", index=False)
    pd.DataFrame(transfer).to_csv(out / "donor_transfer.tsv", sep="\t", index=False)
    pd.DataFrame(consistency).to_csv(out / "donor_marker_consistency.tsv", sep="\t", index=False)
    pd.DataFrame(slice_transfer).to_csv(out / "slice_transfer.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
