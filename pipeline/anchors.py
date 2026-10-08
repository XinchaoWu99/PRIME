"""Per-projection MNN votes, consensus anchors and anchor-level metrics.

`projection_votes` repeats the loop of `prime.core.build_consensus_graph` (projection t uses
GaussianRandomProjection(random_state=seed + t) on L2-normalised X) but keeps every projection's MNN set.
A (K, tau) consensus with K <= K_max is then a slice of the same votes and equals a fresh PRIME run with that K.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, identity
from sklearn.preprocessing import normalize
from sklearn.random_projection import GaussianRandomProjection

import common  # noqa: F401  (makes the prime package of this checkout importable)
from prime import core as prime_core


def projection_votes(X, batch: np.ndarray, n_proj: int, dim: int, k: int, seed: int) -> dict:
    """Unordered cross-batch MNN pairs (i < j) and a pairs x projections 0/1 vote matrix."""
    n = X.shape[0]
    X_norm = normalize(X, axis=1)
    keys, secs = [], []
    for t in range(n_proj):
        t0 = time.perf_counter()
        Xp = GaussianRandomProjection(n_components=dim, random_state=seed + t).fit_transform(X_norm)
        r, c = (np.asarray(a, dtype=np.int64) for a in prime_core.find_multibatch_mnn_graph(Xp, batch, k))
        keys.append(np.unique(r[r < c] * n + c[r < c]))
        secs.append(time.perf_counter() - t0)
    pairs = np.unique(np.concatenate(keys))
    rows = np.concatenate([np.searchsorted(pairs, kk) for kk in keys])
    cols = np.repeat(np.arange(n_proj), [len(kk) for kk in keys])
    V = csr_matrix((np.ones(len(rows), np.int32), (rows, cols)), shape=(len(pairs), n_proj))
    return {"i": pairs // n, "j": pairs % n, "V": V, "seconds": np.array(secs), "n": n}


def min_votes(K: int, tau: float) -> int:
    """Smallest vote count c with c / K >= tau, i.e. the rule applied by PRIME."""
    return next((c for c in range(1, K + 1) if c / K >= tau), K + 1)


def consensus(votes: dict, K: int, tau: float):
    """Anchors kept by the first K projections at threshold tau, with weight = votes / K."""
    cnt = np.asarray(votes["V"][:, :K].sum(axis=1)).ravel()
    keep = cnt >= min_votes(K, tau)
    return votes["i"][keep], votes["j"][keep], cnt[keep] / K


def correct(adata, i, j, w, sigma: float, smoothing: bool = True, chunk_size: int = 2000) -> np.ndarray:
    """Run PRIME's own correction step (`ensemble_mnn_correct`) on a given anchor graph.

    The graph builder is swapped for the precomputed graph; `smoothing=False` swaps
    the kNN smoothing kernel for the identity (module ablation). Nothing else changes.
    """
    n = adata.n_obs
    if len(i) == 0:                       # prime returns the input unchanged in this case
        X = adata.X
        return X.toarray() if hasattr(X, "toarray") else np.array(X)
    G = csr_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n), dtype=np.float32)
    saved = prime_core.build_consensus_graph, prime_core._compute_smoothing_matrix
    prime_core.build_consensus_graph = lambda *a, **kw: G
    if not smoothing:
        prime_core._compute_smoothing_matrix = lambda X, sigma=1.0, k_smooth=15: identity(n, format="csr")
    try:
        return prime_core.ensemble_mnn_correct(adata, "batch", sigma=sigma, inplace=False, chunk_size=chunk_size)
    finally:
        prime_core.build_consensus_graph, prime_core._compute_smoothing_matrix = saved


def searched_batch_pairs(batch: np.ndarray, strategy: str = "pairwise") -> list[tuple]:
    """Batch pairs in which PRIME searches MNNs ('star' = largest batch vs the rest, as prime_st)."""
    b, n = np.unique(batch, return_counts=True)
    if strategy == "star":
        ref = b[np.argmax(n)]
        return [(ref, x) for x in b if x != ref]
    return [(b[p], b[q]) for p in range(len(b)) for q in range(p + 1, len(b))]


def anchor_metrics(i, j, labels, batch, strategy: str = "pairwise"):
    """Precision / FDR / proxy TPR / FPR / coverage of accepted anchors.

    Denominators run over *all* cross-batch pairs in the searched batch pairs
    (pairs of labelled cells only), not over the pairs that some projection found.
    """
    lab, bat = np.asarray(labels).astype(str), np.asarray(batch).astype(str)
    ok = lab != "nan"
    keep = ok[i] & ok[j]
    i, j = i[keep], j[keep]
    y = lab[i] == lab[j]
    ct = pd.crosstab(bat[ok], lab[ok])
    P = sum(float(ct.loc[a] @ ct.loc[b]) for a, b in searched_batch_pairs(bat[ok], strategy))
    T = sum(float(ct.loc[a].sum() * ct.loc[b].sum()) for a, b in searched_batch_pairs(bat[ok], strategy))
    tp, fp = int(y.sum()), int((~y).sum())

    covered = np.zeros(len(lab), bool)
    covered[i[y]] = covered[j[y]] = True
    cov = (pd.DataFrame({"batch": bat[ok], "label": lab[ok], "covered": covered[ok]})
           .groupby(["batch", "label"], observed=True)["covered"].agg(["mean", "size"]).reset_index())
    out = {
        "n_anchors": tp + fp,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "fdr": fp / (tp + fp) if tp + fp else np.nan,
        "proxy_tpr": tp / P if P else np.nan,
        "fpr": fp / (T - P) if T - P else np.nan,
        "coverage": covered[ok].mean(),
        "coverage_min_group": cov["mean"].min(),
    }
    return out, cov


def precision_ci(i, y, B: int = 500, seed: int = 0):
    """95% CI of anchor precision, resampling anchor cells (pairs sharing a cell move together)."""
    cells, idx = np.unique(i, return_inverse=True)
    tp = np.bincount(idx, weights=y.astype(float), minlength=len(cells))
    tot = np.bincount(idx, minlength=len(cells)).astype(float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(cells), len(cells)), minlength=len(cells))
        boots.append(w @ tp / (w @ tot))
    return np.percentile(boots, [2.5, 97.5])


def poisson_binomial_tail(p: np.ndarray, m: int) -> float:
    """P(sum of independent Bernoulli(p_t) >= m)."""
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for q in p:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= 1 - q
    return float(dist[m:].sum())


def vote_structure(votes: dict, K: int, taus, labels, batch, strategy: str = "pairwise", only_pair=None):
    """Condorcet check on the first K projections, separately for same-type (pos) and
    different-type (neg) candidate pairs.

    Returns (summary rows per class x tau, phi correlation matrices per class,
    vote-fraction histogram of pairs found by >= 1 projection). Pairs never found
    by any projection have all-zero votes, so rates and correlations are exact
    over the full candidate set without enumerating it. `only_pair=(a, b)` restricts
    everything to one batch pair (stratified analysis).
    """
    lab, bat = np.asarray(labels).astype(str), np.asarray(batch).astype(str)
    ok = lab != "nan"
    i, j, V = votes["i"], votes["j"], votes["V"][:, :K]
    keep = ok[i] & ok[j]
    pairs = searched_batch_pairs(bat[ok], strategy)
    if only_pair is not None:
        pairs = [tuple(only_pair)]
        keep &= np.isin(bat[i], only_pair) & np.isin(bat[j], only_pair) & (bat[i] != bat[j])
    y = (lab[i] == lab[j])[keep]
    V = V[keep].astype(np.float64)
    ct = pd.crosstab(bat[ok], lab[ok])
    P = sum(float(ct.loc[a] @ ct.loc[b]) for a, b in pairs)
    T = sum(float(ct.loc[a].sum() * ct.loc[b].sum()) for a, b in pairs)

    rows, phis, hist = [], {}, []
    for cls, mask, total in (("pos", y, P), ("neg", ~y, T - P)):
        if total == 0:          # e.g. 293T-only x Jurkat-only batches have no same-type pairs
            continue
        Vc = V[mask]
        a = np.asarray(Vc.sum(axis=0)).ravel()
        both = (Vc.T @ Vc).toarray()
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = (total * both - np.outer(a, a)) / np.sqrt(np.outer(a * (total - a), a * (total - a)))
        phi[~np.isfinite(phi)] = np.nan      # zero-variance projections -> NA, not 0
        phis[cls] = phi
        off = phi[~np.eye(K, dtype=bool)]
        cnt = np.asarray(Vc.sum(axis=1)).ravel()
        hist.append(pd.DataFrame({"class": cls, "votes": np.arange(1, K + 1),
                                  "n_pairs": np.bincount(cnt.astype(int), minlength=K + 1)[1:]}))
        for tau in taus:
            m = min_votes(K, tau)
            rows.append({
                "class": cls, "K": K, "tau": tau, "min_votes": m, "n_candidates": total,
                "mean_accept_rate": a.mean() / total,
                "observed_consensus_rate": (cnt >= m).sum() / total,
                "independent_consensus_rate": poisson_binomial_tail(a / total, m),
                "mean_phi": np.nanmean(off) if K > 1 else np.nan,
                "frac_found_majority_accept": (cnt > K / 2).mean() if len(cnt) else np.nan,
            })
    return pd.DataFrame(rows), phis, pd.concat(hist)
