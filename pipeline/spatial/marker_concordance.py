"""Marker-gene and layer-marker analysis on PRIME's DLPFC domains, compared with the manual layers.

    python marker_concordance.py

The same downstream analysis is run twice on the same spots: once with the manual layers (ground truth, GT)
and once with PRIME's spatial domains (GMM, K = 7, from domain_validation.py), and the two results are
compared. Only spots with a manual layer are used (47,329 of 47,681).

Domains are matched one-to-one to layers once for all 12 sections (Hungarian assignment on the spot overlap):
PRIME integrates the sections, so a domain has the same identity in every section. The matching uses the labels
and is for evaluation only.

Differential expression (both labelings, identical procedure): counts are summed per section x group
(pseudobulk; groups with < 10 spots in a section dropped), log2(CPM + 1), and for every group an enrichment
model  logCPM ~ [group == g] + section  is fitted by least squares (the section fixed effects remove section /
donor differences; "rest" = the other groups of the same sections). t statistics, log2 fold changes and
Benjamini-Hochberg FDR per group. Gene universe: pooled CPM >= 1 over all labelled spots (label independent).

Inputs: out_dir/domains/<tag>/domains_PRIME_s<seed>.tsv for the tags `emb_s0..4` (default setting) and
`tuned_emb_s0..4` (selected setting), i.e. domain_validation.py run once per embedding seed.

Outputs (out_dir/marker_concordance):
  gt_enrichment.tsv                   gene x (t, logFC, fdr) for the seven manual layers
  runs.tsv                            one row per PRIME run (default / selected setting x 5 embedding seeds
                                      x 3 GMM seeds): ARI, NMI, matched accuracy, adjacent-layer share of errors,
                                      marker-programme correlation and top-100 marker overlap, layers whose majority
                                      domain is the same in every donor, pre-specified marker peaks
  per_layer.tsv                       per layer and run: one-to-one matched domain (precision / recall / F1, t
                                      correlation, top-100 overlap) and best-correlated domain; donor consistency
  per_section.tsv                     per section and run: ARI, matched accuracy
  donor_consistency.tsv               per layer, donor and run: majority domain and its share
  showcase/                           one example run (selected setting, embedding seed 0, GMM seed 0):
                                      prime_enrichment.tsv, confusion.tsv, matching.tsv, tcor.tsv, spots.tsv.gz,
                                      dotplot.tsv, marker_sets.tsv, gt_top100.tsv, prime_top100.tsv

Marker lists: FDR < 0.05 and logFC > 0, top 100 by t. Pre-specified layer markers: markers_dlpfc.tsv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import common as C
from prime.metrics.xlc import LAYER_TO_RANK  # noqa: E402

LAYERS = list(LAYER_TO_RANK)                       # Layer1 .. Layer6, WM
MIN_SPOTS = 10                                     # pseudobulk group kept if >= 10 spots in the section
TOP = 100                                          # marker list length for the overlap
RUNS = [(setting, f"{prefix}emb_s{e}", g) for setting, prefix in (("default", ""), ("selected", "tuned_"))
        for e in range(5) for g in range(3)]
SHOWCASE = ("selected", "tuned_emb_s0", 0)         # the example run written to showcase/


def bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def enrichment(X: sp.csr_matrix, groups: np.ndarray, section: np.ndarray, genes: np.ndarray, keep: np.ndarray):
    """Pseudobulk enrichment statistics (see module docstring). Returns {group: DataFrame(t, logFC, fdr)}."""
    key = pd.Series(section).astype(str) + "|" + pd.Series(groups).astype(str)
    codes, uniq = pd.factorize(key)
    n_spots = np.bincount(codes)
    A = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
    pb = np.asarray((A @ X).todense())                                      # pseudobulk x all genes
    ok = n_spots >= MIN_SPOTS
    pb, uniq = pb[ok], np.asarray(uniq)[ok]
    lib = pb.sum(1, keepdims=True)
    Y = np.log2(pb[:, keep] / lib * 1e6 + 1)
    sec = np.array([u.split("|")[0] for u in uniq])
    grp = np.array([u.split("|")[1] for u in uniq])
    S = pd.get_dummies(sec).to_numpy(float)
    res = {}
    for g in np.unique(grp):
        Xd = np.column_stack([(grp == g).astype(float), S])
        XtXi = np.linalg.pinv(Xd.T @ Xd)
        beta = XtXi @ Xd.T @ Y
        r = Y - Xd @ beta
        df = len(Y) - np.linalg.matrix_rank(Xd)
        s2 = (r ** 2).sum(0) / df
        t = beta[0] / np.sqrt(s2 * XtXi[0, 0] + 1e-300)
        p = 2 * stats.t.sf(np.abs(t), df)
        res[g] = pd.DataFrame({"t": t, "logFC": beta[0], "p": p, "fdr": bh(p)}, index=genes[keep])
    return res


def top_markers(tab: pd.DataFrame, n=TOP) -> list[str]:
    """Enriched genes (FDR < 0.05, logFC > 0), top n by t."""
    s = tab[(tab.fdr < 0.05) & (tab.logFC > 0)]
    return s.sort_values("t", ascending=False).index[:n].tolist()


def match(gt: np.ndarray, dom: np.ndarray):
    """One-to-one domain -> layer assignment maximising the spot overlap; unmatched domains map to None."""
    ct = pd.crosstab(pd.Series(dom, name="domain"), pd.Series(gt, name="layer")).reindex(columns=LAYERS, fill_value=0)
    r, c = linear_sum_assignment(-ct.to_numpy())
    m = {ct.index[i]: ct.columns[j] for i, j in zip(r, c)}
    return m, ct


def main():
    cfg = C.load_config()
    OUT = Path(cfg["out_dir"])
    out = OUT / "marker_concordance"
    (out / "showcase").mkdir(parents=True, exist_ok=True)

    adata = C.load_dataset("dlpfc", cfg)
    lab = adata.obs["label"].astype(str).to_numpy()
    adata = adata[np.isin(lab, LAYERS)].copy()
    gt = adata.obs["label"].astype(str).to_numpy()
    section = adata.obs["batch"].astype(str).to_numpy()
    donor = adata.obs["donor"].astype(str).to_numpy()
    X = sp.csr_matrix(adata.X, dtype=np.float64)
    genes = adata.var_names.to_numpy()
    cpm = np.asarray(X.sum(0)).ravel() / X.sum() * 1e6
    keep = cpm >= 1
    print(f"{adata.n_obs} labelled spots, {keep.sum()} / {len(genes)} genes in the universe (pooled CPM >= 1)", flush=True)

    gt_res = enrichment(X, gt, section, genes, keep)
    pd.concat({L: gt_res[L][["t", "logFC", "fdr"]] for L in LAYERS}, axis=1).to_csv(out / "gt_enrichment.tsv", sep="\t")
    gt_top = {L: top_markers(gt_res[L]) for L in LAYERS}
    print("GT enriched genes (FDR<0.05, logFC>0):",
          {L: int(((gt_res[L].fdr < 0.05) & (gt_res[L].logFC > 0)).sum()) for L in LAYERS}, flush=True)

    frozen = pd.read_csv(Path(__file__).resolve().parent / cfg["domains"]["markers"], sep="\t")
    frozen = frozen[frozen.gene.isin(genes[keep])]
    # log-normalised expression (CP10k) of the genes drawn in the dot plot
    ln = adata.copy()
    sc.pp.normalize_total(ln, target_sum=1e4)
    sc.pp.log1p(ln)
    FZ = np.asarray(ln[:, frozen.gene.tolist()].X.todense())

    runs, per_layer, per_section, donor_cons = [], [], [], []
    for setting, emb, gseed in RUNS:
        f = OUT / "domains" / emb / f"domains_PRIME_s{gseed}.tsv"
        d = pd.read_csv(f, sep="\t", index_col=0)["gmm"].reindex(adata.obs_names)
        ok = d.notna().to_numpy()
        dom = d.fillna(-1).astype(int).astype(str).to_numpy()
        m, ct = match(gt[ok], dom[ok])
        mapped = np.array([m.get(x) for x in dom], dtype=object)
        rank_gt = np.array([LAYER_TO_RANK[x] for x in gt])
        rank_pr = np.array([LAYER_TO_RANK.get(x, np.nan) if x else np.nan for x in mapped], float)
        err = ok & (mapped != gt)

        pr_res = enrichment(X[ok], dom[ok], section[ok], genes, keep)
        pr_top = {dm: top_markers(pr_res[dm]) for dm in pr_res}
        inv = {v: k for k, v in m.items()}
        # domain x layer correlation of enrichment t statistics over the gene universe (spatialLIBD-style)
        tcor = pd.DataFrame({L: {dm: np.corrcoef(pr_res[dm]["t"], gt_res[L]["t"])[0, 1] for dm in pr_res}
                             for L in LAYERS})
        dom_layer = tcor.idxmax(axis=1)                      # expression-based identity of every domain
        row = {"setting": setting, "emb": emb, "gmm_seed": gseed, "n_spots": int(ok.sum()),
               "ARI": adjusted_rand_score(gt[ok], dom[ok]), "NMI": normalized_mutual_info_score(gt[ok], dom[ok]),
               "accuracy": float((mapped[ok] == gt[ok]).mean()),
               "adjacent_share_of_errors": float((np.abs(rank_pr[err] - rank_gt[err]) == 1).mean())}
        for L in LAYERS:
            dm = inv.get(L)                                  # one-to-one matched domain (spot overlap)
            bd = tcor[L].idxmax()                            # domain whose marker programme is closest to L
            prec = float(((mapped == L) & (gt == L))[ok].sum() / max((mapped[ok] == L).sum(), 1))
            recl = float(((mapped == L) & (gt == L))[ok].sum() / max((gt[ok] == L).sum(), 1))
            ov = lambda x: len(set(pr_top[x]) & set(gt_top[L])) / max(len(gt_top[L]), 1)
            cons = {}
            for dn in np.unique(donor):                      # majority domain of layer L within each donor
                k = ok & (gt == L) & (donor == dn)
                if k.sum() >= 50:
                    vc = pd.Series(dom[k]).value_counts(normalize=True)
                    cons[dn] = vc.index[0]
                    donor_cons.append({"setting": setting, "emb": emb, "gmm_seed": gseed, "layer": L, "donor": dn,
                                       "majority_domain": vc.index[0], "fraction": float(vc.iloc[0])})
            per_layer.append({"setting": setting, "emb": emb, "gmm_seed": gseed, "layer": L,
                              "matched_domain": dm, "precision": prec, "recall": recl,
                              "F1": 2 * prec * recl / max(prec + recl, 1e-12),
                              "matched_t_cor": tcor.loc[dm, L] if dm in tcor.index else np.nan,
                              "matched_top100_recovery": ov(dm) if dm in pr_top else np.nan,
                              "best_domain": bd, "best_t_cor": tcor.loc[bd, L], "best_top100_recovery": ov(bd),
                              "best_domain_shared": int((tcor.idxmax(axis=0) == bd).sum() > 1),
                              "donor_consistent": int(len(set(cons.values())) == 1), "n_donors": len(cons)})
        pl = pd.DataFrame([r for r in per_layer if (r["setting"], r["emb"], r["gmm_seed"]) == (setting, emb, gseed)])
        # pre-specified layer markers: is the gene's peak group its own layer (manual layers), and is the PRIME
        # domain with the highest expression one whose marker programme matches that layer?
        fz_gt, fz_pr = [], []
        for j, L in enumerate(frozen.layer):
            x = FZ[:, j]
            fz_gt.append(pd.Series(x).groupby(gt).mean().idxmax() == L)
            fz_pr.append(dom_layer[pd.Series(x[ok]).groupby(dom[ok]).mean().idxmax()] == L)
        row.update({"mean_best_t_cor": pl.best_t_cor.mean(), "mean_best_top100_recovery": pl.best_top100_recovery.mean(),
                    "mean_matched_t_cor": pl.matched_t_cor.mean(),
                    "layers_donor_consistent": int(pl.donor_consistent.sum()),
                    "frozen_peak_gt": float(np.mean(fz_gt)), "frozen_peak_prime": float(np.mean(fz_pr)),
                    "n_frozen": len(fz_gt), "domain_layers": ",".join(f"{k}:{v}" for k, v in dom_layer.items())})
        runs.append(row)
        for s in np.unique(section):
            k = ok & (section == s)
            per_section.append({"setting": setting, "emb": emb, "gmm_seed": gseed, "section": s,
                                "donor": donor[k][0], "ARI": adjusted_rand_score(gt[k], dom[k]),
                                "accuracy": float((mapped[k] == gt[k]).mean())})
        print(setting, emb, gseed, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)

        if (setting, emb, gseed) == SHOWCASE:
            sh = out / "showcase"
            pd.concat({dm: pr_res[dm][["t", "logFC", "fdr"]] for dm in pr_res}, axis=1).to_csv(sh / "prime_enrichment.tsv", sep="\t")
            ct.to_csv(sh / "confusion.tsv", sep="\t")
            pd.Series(m, name="layer").rename_axis("domain").to_csv(sh / "matching.tsv", sep="\t")
            tcor.rename_axis("domain").to_csv(sh / "tcor.tsv", sep="\t")
            pd.DataFrame({"spot": adata.obs_names[ok], "section": section[ok], "donor": donor[ok], "gt": gt[ok],
                          "domain": dom[ok], "matched_layer": mapped[ok].astype(str)}).to_csv(
                sh / "spots.tsv.gz", sep="\t", index=False)
            pd.DataFrame({"domain": list(pr_top), "top100": [",".join(v) for v in pr_top.values()]}).to_csv(
                sh / "prime_top100.tsv", sep="\t", index=False)
            # dot plot: pre-specified markers + top 3 GT markers per layer, by manual layer and by PRIME domain
            rows = [{"gene": g, "layer": L, "source": "pre-specified"} for g, L in zip(frozen.gene, frozen.layer)]
            for L in LAYERS:
                for g in [g for g in gt_top[L] if g not in {r["gene"] for r in rows}][:3]:
                    rows.append({"gene": g, "layer": L, "source": "top GT marker"})
            ms = pd.DataFrame(rows)
            ms.to_csv(sh / "marker_sets.tsv", sep="\t", index=False)
            E = np.asarray(ln[:, ms.gene.tolist()].X.todense())
            gi = pd.Index(genes).get_indexer(ms.gene)
            dot = []
            for side, lbl, mask, groups in (("GT", gt, np.ones(len(gt), bool), LAYERS),
                                            ("PRIME", dom, ok, sorted(pr_res, key=int))):
                for gname in groups:
                    k = mask & (lbl == gname)
                    tot = np.asarray(X[k].sum(0)).ravel()           # pooled pseudobulk, same scale as the DE
                    pb = np.log2(tot[gi] / tot.sum() * 1e6 + 1)
                    for j, g in enumerate(ms.gene):
                        dot.append({"side": side, "group": gname, "gene": g, "mean": E[k, j].mean(),
                                    "pb_logcpm": pb[j], "frac": (E[k, j] > 0).mean(), "n": int(k.sum())})
            pd.DataFrame(dot).to_csv(sh / "dotplot.tsv", sep="\t", index=False)
            pd.DataFrame([{"layer": L, "gene": g, "rank": i + 1} for L in LAYERS for i, g in enumerate(gt_top[L])]).to_csv(
                sh / "gt_top100.tsv", sep="\t", index=False)

    pd.DataFrame(runs).to_csv(out / "runs.tsv", sep="\t", index=False)
    pd.DataFrame(per_layer).to_csv(out / "per_layer.tsv", sep="\t", index=False)
    pd.DataFrame(per_section).to_csv(out / "per_section.tsv", sep="\t", index=False)
    pd.DataFrame(donor_cons).to_csv(out / "donor_consistency.tsv", sep="\t", index=False)
    r = pd.DataFrame(runs)
    print(r.select_dtypes("number").groupby(r.setting).agg(["mean", "std"]).T.round(3).to_string(), flush=True)
    print(pd.DataFrame(per_layer).groupby(["setting", "layer"]).mean(numeric_only=True).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
