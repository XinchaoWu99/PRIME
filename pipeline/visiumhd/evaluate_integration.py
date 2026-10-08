"""Score integrated embeddings of a multi-sample Visium HD data set.

    python evaluate_integration.py DATASET METHOD=emb.npy [METHOD=emb.npy ...] --labels-tsv LABELS [--tag s0]

DATASET is `brain_dg_tiles`, `brain_cortex_tiles` or `brain_sections`; each .npy has one row per bin in file
order (as written by spatial/run_methods.py). Labels: --labels-tsv gives the region / layer labels of
annotate_regions.py (key -> label_region), scored on the regions with >= 20 bins in every sample; without it the
obs column named in config.yaml (`datasets.<DATASET>.label`, or --label-col) is used. Unlabelled bins are
integrated but not scored, and every method is scored on the bins that all methods returned.
Outputs (out_dir/visiumhd/<DATASET>/eval/<tag>/):
  scib.tsv        fixed scIB metric set on (up to) 50k labelled bins, stratified by sample x label
  per_method.tsv  - cross-sample label transfer (kNN, fit on all other samples, predict the held-out one),
                    restricted to labels present in all samples (transfer_accuracy, transfer_macro_f1)
                  - cross-sample neighbour fraction (15-NN) over the labels shared by all samples (higher =
                    samples aligned) with its fully-mixed expectation (cross_patient_nn_shared_nonmalignant*).
                    Labels containing "tumor" are treated as sample-specific and reported separately
                    (cross_patient_nn_tumour*; lower = kept apart); the mouse brain data has none
                  - physical-neighbour retention (8 spatial NN also embedding NN, per sample)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import common as C


def knn(X, k, query=None):
    nn = NearestNeighbors(n_neighbors=k + (query is None), n_jobs=16).fit(X)
    idx = nn.kneighbors(X if query is None else query, return_distance=False)
    return idx[:, 1:] if query is None else idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["brain_dg_tiles", "brain_cortex_tiles", "brain_sections"])
    ap.add_argument("embeddings", nargs="+", help="METHOD=path.npy")
    ap.add_argument("--tag", default="s0")
    ap.add_argument("--label-col", default=None, help="score on another obs column (e.g. label_class)")
    ap.add_argument("--labels-tsv", default=None,
                    help="labels from annotate_regions.py apply (key -> label_region, or the column given by --label-col); "
                         "scored on the regions with >= 20 bins in every sample of this dataset")
    args = ap.parse_args()
    cfg = C.load_config()
    if args.label_col and not args.labels_tsv:
        cfg["datasets"][args.dataset]["label"] = args.label_col
    out = Path(cfg["out_dir"]) / "visiumhd" / args.dataset / "eval" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    adata = C.load_dataset(args.dataset, cfg)
    if args.labels_tsv:
        reg = pd.read_csv(args.labels_tsv, sep="\t", index_col=0)[args.label_col or "label_region"].reindex(adata.obs_names)
        reg = reg.fillna("nan").astype(str)
        n = pd.crosstab(reg[reg != "nan"], adata.obs["batch"][reg != "nan"])
        ok = n.index[(n >= 20).all(1)]
        print("regions scored:", n.loc[ok].to_dict("index"), "\nnot scored:", list(n.index.difference(ok)), flush=True)
        adata.obs["label"] = reg.where(reg.isin(ok), "nan").astype("category")
    embs = {m: np.load(p) for m, p in (e.split("=", 1) for e in args.embeddings)}
    keep = np.all([np.isfinite(Z).all(1) for Z in embs.values()], axis=0)
    print(f"{keep.sum()} / {len(keep)} bins returned by all methods", flush=True)
    obs = adata.obs.loc[keep, ["batch", "label"]].astype(str)
    xy = np.asarray(adata.obsm["spatial"], float)[keep]
    embs = {m: C.eval_view(Z[keep], cfg["eval_pcs"]) for m, Z in embs.items()}

    lab, pat = obs["label"].to_numpy(), obs["batch"].to_numpy()
    single = lab != "nan"
    counts = pd.crosstab(lab[single], pat[single])
    shared = counts.index[(counts >= 20).all(1)]
    tumour = np.array(["tumor" in x.lower() for x in lab])
    shared_nm = single & np.isin(lab, shared) & ~tumour

    # scIB on a stratified sample of labelled bins
    s_idx = np.flatnonzero(single)
    s_idx = np.sort(s_idx[C.stratified_order(obs.iloc[s_idx])[:50000]])
    ev = C.sc.AnnData(obs=obs.iloc[s_idx].astype("category"), obsm={m: Z[s_idx] for m, Z in embs.items()})
    res = C.integration_metrics(ev, list(embs), n_jobs=16)
    res.to_csv(out / "scib.tsv", sep="\t")

    rows = []
    for m, Z in embs.items():
        r = {"method": m, "n_bins": int(keep.sum()), "n_shared_types": len(shared)}
        # cross-sample label transfer on shared labels
        accs, f1s = [], []
        for p in np.unique(pat):
            tr, te = single & np.isin(lab, shared) & (pat != p), single & np.isin(lab, shared) & (pat == p)
            pred = KNeighborsClassifier(15, n_jobs=16).fit(Z[tr], lab[tr]).predict(Z[te])
            accs.append((pred == lab[te]).mean())
            f1s.append(f1_score(lab[te], pred, average="macro"))
        r.update(transfer_accuracy=np.mean(accs), transfer_macro_f1=np.mean(f1s))
        # cross-sample neighbour fractions among labelled bins
        si = np.flatnonzero(single)
        nb = knn(Z[si], 15)
        other = (pat[si][nb] != pat[si][:, None]).mean(1)
        in_si = {i: n for n, i in enumerate(si)}
        for name, mask in (("shared_nonmalignant", shared_nm), ("tumour", single & tumour)):
            rows_m = [in_si[i] for i in np.flatnonzero(mask)]
            expected = 1 - pd.Series(pat[mask]).value_counts(normalize=True).pow(2).sum()   # fully mixed
            r[f"cross_patient_nn_{name}"] = other[rows_m].mean() if rows_m else np.nan
            r[f"cross_patient_nn_{name}_mixed_expectation"] = expected
        # physical-neighbour retention per sample
        ret = []
        for p in np.unique(pat):
            k = pat == p
            a, b = knn(xy[k], 8), knn(Z[k], 8)
            ret.append(np.mean([len(np.intersect1d(x, y)) / 8 for x, y in zip(a, b)]))
        r["neighbour_retention"] = np.mean(ret)
        rows.append(r)
        print(r, flush=True)
    pd.DataFrame(rows).to_csv(out / "per_method.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
