"""Anchor composition and local structure of the ablation variants on the human immune data.

    python anchor_diagnosis.py        -> out_dir/ablation/immune_diag/

Compares the full consensus with the no-consensus and single-projection variants. Anchors are re-derived from
the votes the ablation used (anchors.projection_votes, seeds of config.yaml); embeddings are the saved outputs
of `parameter_grid.py immune ablation`.
  anchor_type_pairs.tsv   anchors per (seed, variant, type pair)
  coverage_by_type.tsv    same-type anchor coverage per cell type
  cluster_by_type.tsv     per type: best KMeans F1, 15-NN same-type fraction, 15-NN same-type other-batch fraction
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import anchors as A  # noqa: E402
import common as C  # noqa: E402

cfg = C.load_config()
OUT = cfg["out_dir"]
D = OUT + "/ablation/immune_diag/"
os.makedirs(D, exist_ok=True)
adata = C.load_dataset("immune", cfg)
X, batch, lab = adata.X, adata.obs["batch"].astype(str).values, adata.obs["label"].astype(str).values
types = np.unique(lab); n = len(lab)
P = cfg["datasets"]["immune"]["default"]
variants = {"full": (P["n_projections"], P["consensus_threshold"]), "no_consensus": (P["n_projections"], 1e-9),
            "single_projection": (1, P["consensus_threshold"])}
pair_rows, cov_rows, clu_rows = [], [], []
for seed in cfg["seeds"]:
    votes = A.projection_votes(X, batch, P["n_projections"], P["target_dim"], P["k_neighbors"], seed)
    for v, (K, tau) in variants.items():
        i, j, w = A.consensus(votes, K, tau)
        a, b = np.minimum(lab[i], lab[j]), np.maximum(lab[i], lab[j])
        pc = pd.Series(list(zip(a, b))).value_counts()
        for (x, y), c in pc.items():
            pair_rows.append(dict(seed=seed, variant=v, type_a=x, type_b=y, same=x == y, n=c))
        ok = lab[i] == lab[j]
        has = np.zeros(n, bool); has[i[ok]] = True; has[j[ok]] = True
        anyn = np.zeros(n, bool); anyn[i] = True; anyn[j] = True
        for t in types:
            m = lab == t
            cov_rows.append(dict(seed=seed, variant=v, type=t, n_cells=m.sum(), coverage_same=has[m].mean(),
                                 any_anchor=anyn[m].mean(), n_batches=len(set(batch[m]))))
        Z = C.eval_view(np.load(f"{OUT}/ablation/immune/embeddings/{v}_s{seed}.npy"), cfg["eval_pcs"], seed)
        km = KMeans(len(types), n_init=10, random_state=0).fit_predict(Z)
        nb = NearestNeighbors(n_neighbors=16).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
        for t in types:
            m = lab == t
            f1 = max(2 * (m & (km == c)).sum() / (m.sum() + (km == c).sum()) for c in np.unique(km[m]))
            clu_rows.append(dict(seed=seed, variant=v, type=t, kmeans_best_f1=f1,
                                 nn_same_type=(lab[nb[m]] == t).mean(),
                                 nn_other_batch_same_type=((lab[nb[m]] == t) & (batch[nb[m]] != batch[m][:, None])).mean()))
    print("seed", seed, "done", flush=True)
P = pd.DataFrame(pair_rows); P.to_csv(D + "anchor_type_pairs.tsv", sep="\t", index=False)
V = pd.DataFrame(cov_rows); V.to_csv(D + "coverage_by_type.tsv", sep="\t", index=False)
K = pd.DataFrame(clu_rows); K.to_csv(D + "cluster_by_type.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
wrong = P[~P.same].groupby(["variant", "type_a", "type_b"]).n.mean().unstack(0).fillna(0)
wrong["extra_nocons_vs_full"] = wrong["no_consensus"] - wrong["full"]
print(wrong.sort_values("extra_nocons_vs_full", ascending=False).head(15).round(0))
print(V.groupby(["type", "variant"]).coverage_same.mean().unstack().round(3))
k = K.groupby(["type", "variant"])[["kmeans_best_f1", "nn_same_type", "nn_other_batch_same_type"]].mean().unstack().round(3)
print(k)
