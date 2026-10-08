from __future__ import annotations

# ---- Standard / typing ----
from inspect import signature
from typing import Optional, Tuple, Dict, Any, Union

# ---- Numeric / sparse ----
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, issparse, diags, identity
from scipy.sparse.linalg import cg, LinearOperator

# ---- ML utilities ----
from sklearn.preprocessing import normalize
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

# ---- Single-cell ----
import scanpy as sc
from anndata import AnnData

from prime.core import build_consensus_graph

# Default regularisation weights of each integration mode (used when the
# corresponding argument of ``prime_st`` is left at None).
_MODE_DEFAULTS = {
    "laplacian": {"lambda_anchor": 5.0, "lambda_spatial": 1.0},
    "transport": {"lambda_anchor": 0.0, "lambda_spatial": 0.0},
    "fused": {"lambda_anchor": 10.0, "lambda_spatial": 0.1},
}


# ============================================================
# 1. Low-level utilities
# ============================================================

def _as_csr(X) -> csr_matrix:
    """Make sure X is a CSR sparse matrix."""
    return X.tocsr() if issparse(X) else csr_matrix(X)


def _row_norm_log1p(X, target_sum: float = 1e4) -> csr_matrix:
    """
    Library-size normalize each cell (row) to `target_sum`, then apply log1p.
    Equivalent to scanpy.pp.normalize_total + scanpy.pp.log1p, but kept sparse.
    """
    X = _as_csr(X).astype(np.float32)
    rs = np.asarray(X.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0  # avoid divide-by-zero for empty cells
    scale = (target_sum / rs).astype(np.float32)
    X = X.multiply(scale[:, None])
    X.data = np.log1p(X.data).astype(np.float32)
    return X.tocsr()


def _pick_hvgs(
    adata: AnnData,
    *,
    batch_key: str,
    n_hvg: int,
    flavor: str = "seurat_v3",
    layer: Optional[str] = None,
) -> np.ndarray:
    """Batch-aware HVG selection. Writes adata.var['highly_variable']."""
    X_for_hvg = adata.layers[layer] if (layer is not None and layer in adata.layers) else adata.X
    tmp = AnnData(
        X=_as_csr(X_for_hvg),
        obs=adata.obs[[batch_key]].copy(),
        var=adata.var.copy(),
    )
    sc.pp.highly_variable_genes(
        tmp,
        batch_key=batch_key,
        n_top_genes=n_hvg,
        flavor=flavor,
        subset=False,
        inplace=True,
    )
    adata.var["highly_variable"] = tmp.var["highly_variable"].values
    return adata.var["highly_variable"].values.astype(bool)


def _svd_embedding(X: csr_matrix, n_comps: int, random_state: int = 0) -> np.ndarray:
    """Sparse-friendly TruncatedSVD embedding (essentially PCA without centering)."""
    svd = TruncatedSVD(n_components=n_comps, random_state=random_state)
    Z = svd.fit_transform(X)
    return Z.astype(np.float32)


def _random_projection_embedding(
    X,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    """Gaussian random-projection embedding.

    A single dense embedding produced by multiplying X with a Gaussian random
    matrix R (entries ~ N(0, 1 / n_components), as drawn by scikit-learn). By
    the Johnson-Lindenstrauss lemma this approximately preserves pairwise
    Euclidean distances, which is exactly the property distance-based methods
    such as MNN rely on. Sparse-friendly: ``fit_transform`` accepts a CSR X and
    returns a dense ``(n_obs, n_components)`` array.
    """
    rp = GaussianRandomProjection(
        n_components=n_components,
        random_state=random_state,
    )
    return rp.fit_transform(X).astype(np.float32)


# ============================================================
# 2. MNN (Mutual Nearest Neighbors) helpers
# ============================================================

def _find_mutual_nn(a_to_b: np.ndarray, b_to_a: np.ndarray) -> np.ndarray:
    """Given kNN index arrays, return mutual NN pairs as (i_in_a, j_in_b)."""
    forward = set()
    for i in range(a_to_b.shape[0]):
        for j in a_to_b[i]:
            forward.add((i, j))

    pairs = []
    for j in range(b_to_a.shape[0]):
        for i in b_to_a[j]:
            if (i, j) in forward:
                pairs.append((i, j))

    return np.asarray(pairs, dtype=np.int64)


def _mnn_between_groups(
    X: np.ndarray,
    idx1: np.ndarray,
    idx2: np.ndarray,
    k: int,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """MNN between two index sets in a shared feature space X (dense)."""
    if len(idx1) < 2 or len(idx2) < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    k12 = min(k, len(idx2))
    k21 = min(k, len(idx1))

    nn2 = NearestNeighbors(n_neighbors=k12, metric="euclidean", n_jobs=n_jobs).fit(X[idx2])
    _, a_to_b = nn2.kneighbors(X[idx1])

    nn1 = NearestNeighbors(n_neighbors=k21, metric="euclidean", n_jobs=n_jobs).fit(X[idx1])
    _, b_to_a = nn1.kneighbors(X[idx2])

    mutual = _find_mutual_nn(a_to_b, b_to_a)
    if mutual.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    r = idx1[mutual[:, 0]]
    c = idx2[mutual[:, 1]]
    return r.astype(np.int64), c.astype(np.int64)


def _multibatch_mnn_edges(
    X: np.ndarray,
    batch_labels: np.ndarray,
    *,
    k: int,
    strategy: str = "star",   # "star" or "pairwise"
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return symmetric (rows, cols) MNN edges across batches."""
    batches = np.unique(batch_labels)

    if strategy not in ("star", "pairwise"):
        raise ValueError("strategy must be 'star' or 'pairwise'")

    if strategy == "star":
        sizes = {b: int(np.sum(batch_labels == b)) for b in batches}
        ref = max(sizes, key=sizes.get)
        pairs_to_do = [(ref, b) for b in batches if b != ref]
    else:
        pairs_to_do = []
        for i, b1 in enumerate(batches):
            for b2 in batches[i + 1:]:
                pairs_to_do.append((b1, b2))

    rows, cols = [], []
    for b1, b2 in pairs_to_do:
        idx1 = np.where(batch_labels == b1)[0]
        idx2 = np.where(batch_labels == b2)[0]
        r, c = _mnn_between_groups(X, idx1, idx2, k=k, n_jobs=n_jobs)
        if r.size == 0:
            continue
        # symmetric edges
        rows.append(r); cols.append(c)
        rows.append(c); cols.append(r)

    if not rows:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    return np.concatenate(rows), np.concatenate(cols)


def _build_rp_consensus_mnn_graph(
    X_log_hvg: csr_matrix,
    batch_labels: np.ndarray,
    *,
    n_projections: int,
    rp_dim: int,
    k_mnn: int,
    consensus_threshold: float,
    mnn_strategy: str,
    n_jobs: int,
    random_state: int,
) -> csr_matrix:
    """Ensemble random-projection + consensus-MNN anchor graph.

    This is the reusable implementation of the PRIME "ERP consensus MNN" step,
    the spatial counterpart of :func:`prime.core.build_consensus_graph` and
    :func:`prime.gpu.ensemble._gpu_consensus_graph`. The principle, end to end:

    1. L2-normalize each cell's expression row, so kNN compares *profiles*
       (cosine-like geometry) rather than library size.
    2. Draw ``n_projections`` independent Gaussian random projections, each to
       ``rp_dim`` dimensions, with seeds ``random_state + t``. The
       Johnson-Lindenstrauss lemma guarantees each projection approximately
       preserves pairwise distances, so MNNs found in the low-dimensional
       projection match those in the full space, far more cheaply.
    3. In every projection, find cross-batch mutual nearest neighbours.
    4. Vote: an edge's consensus weight is the fraction of projections in which
       it appeared (``count / n_projections``).
    5. Keep edges whose frequency ``>= consensus_threshold``; symmetrize.

    Returns a symmetric CSR anchor graph with no self-loops, whose edge weights
    are consensus frequencies in ``[consensus_threshold, 1]``.
    """
    n = X_log_hvg.shape[0]

    # 1) L2-normalize rows (keeps the matrix sparse).
    X_norm = normalize(X_log_hvg, axis=1)

    # 2-3) Ensemble of projections; collect cross-batch MNN edges per projection.
    #       Edges are encoded as flat int64 keys (r * n + c) for fast voting.
    keys_all = []
    for t in range(n_projections):
        Xp = _random_projection_embedding(X_norm, rp_dim, random_state + t)
        r, c = _multibatch_mnn_edges(
            Xp, batch_labels,
            k=k_mnn, strategy=mnn_strategy, n_jobs=n_jobs,
        )
        if r.size > 0:
            keys_all.append(r.astype(np.int64) * n + c.astype(np.int64))

    if not keys_all:
        raise RuntimeError(
            "No MNN edges found across projections. "
            "Try increasing k_mnn, lowering consensus_threshold, "
            "or using mnn_strategy='pairwise'."
        )

    # 4) Consensus voting: frequency = (#projections that voted) / n_projections.
    all_keys = np.concatenate(keys_all)
    uniq_keys, counts = np.unique(all_keys, return_counts=True)
    freq = counts.astype(np.float32) / float(n_projections)

    # 5) Threshold and symmetrize.
    keep = freq >= float(consensus_threshold)
    if keep.sum() == 0:
        raise RuntimeError(
            "No consensus MNN edges survived consensus_threshold. "
            "Try lowering consensus_threshold or increasing n_projections."
        )

    rows = (uniq_keys[keep] // n).astype(np.int64)
    cols = (uniq_keys[keep] % n).astype(np.int64)
    w_expr = freq[keep].astype(np.float32)

    Wa = coo_matrix((w_expr, (rows, cols)),
                    shape=(n, n), dtype=np.float32).tocsr()
    Wa.sum_duplicates()
    Wa.setdiag(0.0)
    Wa.eliminate_zeros()
    Wa = (Wa + Wa.T) * 0.5
    Wa.eliminate_zeros()
    return Wa


def _ensemble_anchor_graph(
    adata: AnnData,
    batch_labels: np.ndarray,
    batch_key: str,
    *,
    layer: Optional[str] = None,
    target_sum: float = 1e4,
    n_hvg: int = 2000,
    hvg_flavor: str = "cell_ranger",
    n_projections: int = 4,
    target_dim: int = 128,
    k_neighbors: int = 15,
    consensus_threshold: float = 0.4,
    random_state: int = 0,
) -> csr_matrix:
    """Anchor graph of the expression-only PRIME ensemble.

    The consensus graph of :func:`prime.core.build_consensus_graph`, i.e. the
    anchors :func:`prime.ensemble_mnn_correct` uses for scRNA-seq data, with
    spots treated as cells: log1p CP10k of all genes, batch-aware HVGs, random
    projections + mutual nearest neighbours between *every* pair of batches,
    consensus vote ``>= consensus_threshold``. Unlike
    :func:`_build_rp_consensus_mnn_graph` it does not use the HVG set or the
    star topology of the spatial pipeline.

    Returns a symmetric CSR graph holding cross-batch edges only, with edge
    weight = vote fraction.
    """
    X_base = adata.layers[layer] if (layer is not None and layer in adata.layers) else adata.X
    X_log = _row_norm_log1p(X_base, target_sum=target_sum)
    tmp = AnnData(X=X_log, obs=adata.obs[[batch_key]].copy())
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvg, flavor=hvg_flavor, batch_key=batch_key)
    X_hvg = X_log[:, tmp.var["highly_variable"].values]
    G = build_consensus_graph(X_hvg, batch_labels, None, n_projections, target_dim, k_neighbors,
                              consensus_threshold, random_state).tocoo()
    cross = batch_labels[G.row] != batch_labels[G.col]
    n = adata.n_obs
    Wa = coo_matrix((G.data[cross].astype(np.float32), (G.row[cross], G.col[cross])),
                    shape=(n, n), dtype=np.float32).tocsr()
    Wa.sum_duplicates()
    Wa.setdiag(0.0)
    Wa.eliminate_zeros()
    Wa = (Wa + Wa.T) * 0.5
    Wa.eliminate_zeros()
    return Wa


# ============================================================
# 3. Spatial graph + spatial context
# ============================================================

def _build_spatial_graph(
    spatial_coords: np.ndarray,
    batch_labels: np.ndarray,
    Z_gate: np.ndarray,
    *,
    k_spatial: int = 6,
    n_jobs: int = 1,
    spatial_weight_mode: str = "rbf",   # "rbf" or "binary"
    gate_by_expr: bool = True,
    expr_gate_beta: float = 1.0,
    exclude_self: bool = False,
    eps: float = 1e-8,
) -> csr_matrix:
    """
    Within-batch spatial kNN graph (symmetric).
    Edge weight =  RBF(spatial distance) * RBF(expression distance)^beta

    ``exclude_self=False`` (default): the kNN query returns every spot as its
    own first neighbour, so each spot has ``k_spatial - 1`` other neighbours,
    the bandwidth medians include the zero self-distances, and an edge found
    from both ends carries the sum of both directions.
    ``exclude_self=True``: ``k_spatial`` neighbours besides the spot itself,
    medians over real edges only, and every edge counted once (union of the two
    directions).
    """
    n = spatial_coords.shape[0]
    rows, cols, w_list = [], [], []

    for b in np.unique(batch_labels):
        idx = np.where(batch_labels == b)[0]
        if len(idx) < 2:
            continue
        coords = spatial_coords[idx]
        k = min(k_spatial, len(idx) - 1)
        if k < 1:
            continue

        if exclude_self:
            nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=n_jobs).fit(coords)
            dists, nbrs = nn.kneighbors(coords)
            own = nbrs == np.arange(len(idx))[:, None]
            drop = np.where(own.any(axis=1), own.argmax(axis=1), k)   # duplicated coordinates: drop the farthest
            keep = np.ones_like(nbrs, dtype=bool)
            keep[np.arange(len(idx)), drop] = False
            nbrs, dists = nbrs[keep].reshape(len(idx), k), dists[keep].reshape(len(idx), k)
        else:
            nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=n_jobs).fit(coords)
            dists, nbrs = nn.kneighbors(coords)

        r = np.repeat(idx, k)
        c = idx[nbrs.reshape(-1)]

        if spatial_weight_mode == "binary":
            w_sp = np.ones_like(r, dtype=np.float32)
        else:
            d2 = (dists.reshape(-1).astype(np.float32)) ** 2
            sigma2 = np.median(d2) + eps
            w_sp = np.exp(-d2 / (2.0 * sigma2)).astype(np.float32)

        if gate_by_expr:
            diff = Z_gate[r] - Z_gate[c]
            d2e = np.sum(diff * diff, axis=1).astype(np.float32)
            tau2 = np.median(d2e) + eps
            w_expr = np.exp(-d2e / (2.0 * tau2)).astype(np.float32)
            w_sp = w_sp * (w_expr ** expr_gate_beta)

        rows.append(r); cols.append(c); w_list.append(w_sp)
        if not exclude_self:
            rows.append(c); cols.append(r); w_list.append(w_sp)

    if not rows:
        return csr_matrix((n, n), dtype=np.float32)

    rows = np.concatenate(rows).astype(np.int64)
    cols = np.concatenate(cols).astype(np.int64)
    w = np.concatenate(w_list).astype(np.float32)

    if exclude_self:   # the weight is symmetric in (i, j): the union of both directions counts every edge once
        Ws = coo_matrix((w, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
        Ws = Ws.maximum(Ws.T).tocsr()
        Ws.eliminate_zeros()
        return Ws

    Ws = coo_matrix((w, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    Ws.sum_duplicates()
    Ws.setdiag(0.0)
    Ws.eliminate_zeros()
    Ws = (Ws + Ws.T) * 0.5
    Ws.eliminate_zeros()
    return Ws


def _spatial_context_from_graph(Ws: csr_matrix, Z_ctx: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Spatial-context vector for each spot:
        S = D^{-1} W_s Z_ctx     (row-normalized neighborhood mean)
    Then L2-normalize each row so it acts like a direction in feature space.
    """
    d = np.asarray(Ws.sum(axis=1)).ravel().astype(np.float32)
    d[d == 0] = 1.0
    S = (Ws @ Z_ctx) / d[:, None]
    S = normalize(S, axis=1)
    return S.astype(np.float32)


def _rbf_similarity(A: np.ndarray, rows: np.ndarray, cols: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """RBF similarity between rows A[rows] and A[cols] using median heuristic for sigma^2."""
    diff = A[rows] - A[cols]
    d2 = np.sum(diff * diff, axis=1).astype(np.float32)
    sigma2 = np.median(d2) + eps
    return np.exp(-d2 / (2.0 * sigma2)).astype(np.float32)


def _reweight_anchors_by_spatial_context(
    Wa: csr_matrix,
    Ws: csr_matrix,
    Z_ctx: np.ndarray,
    power: float = 1.0,
) -> csr_matrix:
    """Down-weight anchors that join spots in dissimilar spatial neighbourhoods.

    Every anchor weight is multiplied by ``RBF(S_i, S_j) ** power``, where ``S``
    is the spatial-context vector of :func:`_spatial_context_from_graph`. RBF
    similarity is symmetric, so the reweighted graph remains symmetric and
    self-loop-free.
    """
    n = Wa.shape[0]
    Wa_coo = Wa.tocoo()
    rows = Wa_coo.row.astype(np.int64)
    cols = Wa_coo.col.astype(np.int64)
    w_expr = Wa_coo.data.astype(np.float32)

    S = _spatial_context_from_graph(Ws, Z_ctx)
    w_ctx = _rbf_similarity(S, rows, cols)
    w_anchor = w_expr * (w_ctx ** float(power))

    Wa = coo_matrix((w_anchor, (rows, cols)),
                    shape=(n, n), dtype=np.float32).tocsr()
    Wa.sum_duplicates()
    Wa.setdiag(0.0)
    Wa.eliminate_zeros()
    Wa = (Wa + Wa.T) * 0.5
    Wa.eliminate_zeros()
    return Wa


# ============================================================
# 4. Laplacian + CG solver
# ============================================================

def _laplacian(W: csr_matrix) -> csr_matrix:
    """Unnormalized graph Laplacian L = D - W."""
    d = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    return diags(d, offsets=0, format="csr") - W


def _cg_solve_matrix_rhs(
    A: csr_matrix,
    B: np.ndarray,
    *,
    tol: float = 1e-5,
    maxiter: int = 200,
    x0_zero: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve A X = B for multiple RHS columns using CG per column.
    Cross-version compatible with SciPy (old `tol=` and new `rtol=` APIs).
    Uses Jacobi (diagonal) preconditioner. The initial guess is B (default) or
    zero (``x0_zero=True``, for systems whose solution is not close to B).
    """
    n, k = B.shape

    diagA = A.diagonal().astype(np.float64)
    diagA[diagA == 0] = 1.0
    M = LinearOperator((n, n), matvec=lambda x: x / diagA, dtype=np.float64)

    cg_params = signature(cg).parameters
    use_new_api = ("rtol" in cg_params)

    X = np.zeros_like(B, dtype=np.float64)
    infos = []

    for j in range(k):
        b = B[:, j].astype(np.float64)
        x0 = np.zeros_like(b) if x0_zero else b.copy()
        if use_new_api:
            x, info = cg(A, b, x0=x0, rtol=tol, atol=0.0, maxiter=maxiter, M=M)
        else:
            x, info = cg(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
        X[:, j] = x
        infos.append(int(info))

    stats = {
        "cg_info_per_dim": infos,
        "cg_converged_dims": int(np.sum(np.array(infos) == 0)),
        "cg_failed_dims": int(np.sum(np.array(infos) != 0)),
        "cg_api": "rtol/atol" if use_new_api else "tol",
    }
    return X.astype(np.float32), stats


# ============================================================
# 5. Correction fields over the spatial graph
# ============================================================

def _transport_corrections(
    Z0: np.ndarray,
    Wa: csr_matrix,
    Ws: csr_matrix,
    *,
    step: float = 0.5,
    mu: float = 1e3,
    eps: float = 1e-6,
    tol: float = 1e-6,
    maxiter: int = 2000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Anchor correction vectors, carried to every spot over the spatial graph.

    Every anchored spot ``i`` gets the correction
    ``c_i = step * sum_j w_ij (Z0_j - Z0_i) / sum_j w_ij`` over its cross-batch
    anchors (``step = 0.5``: halfway to the partners, so anchored pairs meet).
    The corrections are then carried to every spot of the same section by
    harmonic interpolation over the within-section spatial graph,

        (L_s + mu * M + eps * I) C = mu * M * C_raw,

    with ``M`` the diagonal indicator of anchored spots: each spot receives a
    spatially smooth average of the corrections of the anchored spots around it.
    """
    n = Z0.shape[0]
    W = Wa.tocoo()
    wsum = np.bincount(W.row, weights=W.data, minlength=n)
    raw = np.zeros_like(Z0, dtype=np.float64)
    np.add.at(raw, W.row, (Z0[W.col] - Z0[W.row]) * W.data[:, None])
    has = wsum > 0
    raw[has] /= wsum[has][:, None]
    raw *= float(step)
    A = (_laplacian(Ws).astype(np.float64) + diags(mu * has.astype(np.float64) + eps)).tocsr()
    C, stats = _cg_solve_matrix_rhs(A, (mu * has[:, None] * raw).astype(np.float32), tol=tol, maxiter=maxiter)
    stats = {f"transport_{k}": v for k, v in stats.items() if k != "cg_info_per_dim"}
    stats["transport_anchored_spots"] = int(has.sum())
    return C, stats


def _unit_mean_degree(W: csr_matrix) -> Tuple[csr_matrix, float]:
    """Scale W so that the mean weighted degree over the spots with at least one edge is 1."""
    d = np.asarray(W.sum(axis=1)).ravel()
    m = float(d[d > 0].mean()) if np.any(d > 0) else 1.0
    return (W * (1.0 / m)).astype(np.float32).tocsr(), m


def _select_lambda_anchor_cv(
    Z0: np.ndarray,
    Wa: csr_matrix,
    Ws: csr_matrix,
    grid=(0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    *,
    holdout: float = 0.2,
    eps: float = 1e-4,
    tol: float = 1e-5,
    maxiter: int = 2000,
    random_state: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """Label-free choice of ``lambda_anchor`` by anchor cross-validation.

    A random ``holdout`` fraction of the anchor pairs is set aside, the
    alignment step of :func:`_fused_graph_regression` is fitted on the remaining
    anchors for every value in ``grid``, and the value with the smallest
    weighted mean squared distance between the held-out partners after
    correction is returned (too small: the section offset is under-corrected;
    too large: the correction follows the noise of single pairs, which does not
    carry over to other pairs). The residual without correction is reported as
    ``"none"``.
    """
    n = Z0.shape[0]
    W = Wa.tocoo()
    up = W.row < W.col
    r, c, w = W.row[up], W.col[up], W.data[up].astype(np.float64)
    rng = np.random.default_rng(random_state)
    test = rng.random(r.size) < holdout
    if test.sum() == 0 or (~test).sum() == 0:
        return float(grid[len(grid) // 2]), {}
    rt, ct, wt = r[~test], c[~test], w[~test]
    Wtr = coo_matrix((np.r_[wt, wt], (np.r_[rt, ct], np.r_[ct, rt])), shape=(n, n)).tocsr()
    La = _laplacian(Wtr).astype(np.float64)
    Ls = _laplacian(Ws).astype(np.float64)
    Z0d = Z0.astype(np.float64)
    rv, cv, wv = r[test], c[test], w[test]

    def resid(Z):
        return float((wv * ((Z[rv] - Z[cv]) ** 2).sum(axis=1)).sum() / wv.sum())

    scores = {"none": resid(Z0d)}
    for lam in grid:
        A = (Ls + lam * La + eps * identity(n, format="csr")).tocsr()
        C, _ = _cg_solve_matrix_rhs(A, (-lam * (La @ Z0d)).astype(np.float32), tol=tol, maxiter=maxiter, x0_zero=True)
        scores[f"{lam:g}"] = resid(Z0d + C)
    best = min(grid, key=lambda lam: scores[f"{lam:g}"])
    return float(best), scores


def _fused_graph_regression(
    Z0: np.ndarray,
    Wa: csr_matrix,
    Ws: csr_matrix,
    batch_labels: np.ndarray,
    *,
    lambda_anchor: float,
    lambda_spatial: float,
    eps: float = 1e-4,
    tol: float = 1e-6,
    maxiter: int = 5000,
    smooth_tol: float = 1e-5,
    smooth_maxiter: int = 200,
    robust_iters: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Alignment and smoothing on one fused graph.

    ``Wa`` (cross-section anchors) and ``Ws`` (within-section spatial graph),
    both scaled to unit mean degree, are fused:

        W_f = W_s + lambda_anchor * W_a,   L_f = L_s + lambda_anchor * L_a.

    Step 1, alignment — Laplacian regression of a correction field C on the
    fused graph:

        min_C  sum_{(i,j) in W_f} w_ij || (c_i - c_j) - t_ij ||^2 + eps ||C||^2,

    with ``t_ij = 0`` on spatial edges (neighbouring spots share the correction)
    and ``t_ij = z0_j - z0_i`` on anchor edges (anchored partners coincide after
    correction), i.e. ``(L_f + eps I) C = -lambda_anchor * L_a Z0``. A shift
    common to a whole section is not penalised by the spatial term, so the
    anchors of a section move it as a whole; ``lambda_anchor`` sets how far the
    correction may bend within a section. Sections without anchors stay put.

    Step 2, smoothing — Laplacian regularisation of the corrected embedding on
    the same graph: ``(I + lambda_spatial * L_f) Z = Z0 + C``.

    ``robust_iters > 0`` reweights step 1 iteratively: after each solve every
    anchor weight is reset to its initial value times
    ``exp(-r_ij^2 / (2 median r^2))``, ``r_ij = ||z_i - z_j||`` after
    correction, so anchors the smooth correction cannot satisfy lose weight; the
    reweighted anchor graph is then used in step 2.
    """
    n = Z0.shape[0]
    Ls = _laplacian(Ws).astype(np.float64)
    W0 = Wa.tocoo()
    ar, ac, w0 = W0.row, W0.col, W0.data.astype(np.float64)
    w = w0.copy()
    for it in range(int(robust_iters) + 1):
        La = _laplacian(coo_matrix((w, (ar, ac)), shape=(n, n)).tocsr()).astype(np.float64)
        Lf = (Ls + lambda_anchor * La).tocsr()
        B = -lambda_anchor * (La @ Z0.astype(np.float64))
        C, st1 = _cg_solve_matrix_rhs((Lf + eps * identity(n, format="csr")).tocsr(), B.astype(np.float32),
                                      tol=tol, maxiter=maxiter, x0_zero=True)
        if it == int(robust_iters):
            break
        Zc = Z0.astype(np.float64) + C
        d2 = ((Zc[ar] - Zc[ac]) ** 2).sum(axis=1)
        w = w0 * np.exp(-d2 / (2.0 * (np.median(d2) + 1e-12)))
    stats = {f"align_{k}": v for k, v in st1.items() if k != "cg_info_per_dim"}
    stats["anchor_weight_retained"] = float(w.sum() / w0.sum()) if w0.sum() > 0 else 0.0
    # share of the correction that is a rigid shift of each section (diagnostic)
    shift = np.zeros_like(C, dtype=np.float64)
    for b in np.unique(batch_labels):
        m = batch_labels == b
        shift[m] = C[m].mean(axis=0)
    tot = float((C.astype(np.float64) ** 2).sum())
    stats.update(correction_rms=float(np.sqrt(tot / n)),
                 correction_section_shift_share=float((shift ** 2).sum() / tot) if tot > 0 else 0.0)
    Z1 = (Z0 + C).astype(np.float32)
    if lambda_spatial > 0:
        A = (identity(n, format="csr") + lambda_spatial * Lf).tocsr()
        Z, st2 = _cg_solve_matrix_rhs(A, Z1, tol=smooth_tol, maxiter=smooth_maxiter)
        stats.update({f"smooth_{k}": v for k, v in st2.items() if k != "cg_info_per_dim"})
    else:
        Z = Z1
    return Z.astype(np.float32), stats


# ============================================================
# 6. Main API: PRIME
# ============================================================

def prime_st(
    adata: AnnData,
    *,
    batch_key: str,
    spatial_key: str = "spatial",
    layer: Optional[str] = None,
    # HVG / normalization
    n_hvg: int = 3000,
    hvg_flavor: str = "seurat_v3",
    target_sum: float = 1e4,
    # Integration mode
    integration: str = "laplacian",       # "laplacian", "transport" or "fused"
    # ERP MNN anchors
    n_projections: int = 10,
    rp_dim: int = 50,
    k_mnn: int = 20,
    consensus_threshold: float = 0.4,
    mnn_strategy: str = "star",          # "star" or "pairwise"
    # Spatial graph
    k_spatial: int = 6,
    gate_spatial_by_expr: bool = True,
    expr_gate_beta: float = 1.0,
    spatial_graph_exclude_self: Optional[bool] = None,   # None: True for "fused", False otherwise
    # Anchor reweighting by spatial context
    reweight_anchors_by_spatial_context: bool = True,
    spatial_power: float = 1.0,
    # Embedding + solver
    n_comps: int = 30,
    svd_dim_for_ctx: int = 50,
    # Projection method for the context (Z_gate) and base (Z0) embeddings
    context_method: str = "svd",            # "svd" or "random_projection"
    base_embedding_method: str = "svd",     # "svd" or "random_projection" (experimental)
    lambda_anchor: Optional[Union[float, str]] = None,   # None: mode default; "auto" (fused only): cross-validation
    lambda_spatial: Optional[float] = None,              # None: mode default
    solver_tol: float = 1e-5,
    solver_maxiter: int = 200,
    # Transport / fused modes
    transport_step: float = 0.5,
    transport_anchor_kwargs: Optional[Dict[str, Any]] = None,
    anchor_source: str = "ensemble",      # fused mode: "ensemble" or "spatial"
    fused_eps: float = 1e-4,
    fused_robust_iters: int = 0,
    # Misc
    n_jobs: int = 1,
    random_state: int = 0,
    key_added: str = "X_prime",
    store_graphs: bool = False,
    graph_key: str = "prime_graphs",
    copy: bool = False,
    verbose: bool = True,
) -> Optional[AnnData]:
    """
    PRIME: Projection-based Robust Integration with Mutual-NN and spatial Embedding.

    Integrate multiple Visium / spatial transcriptomics slices. Every mode
    starts from the same two ingredients:

    * ``Z0`` — the TruncatedSVD embedding of HVG-only log1p data (``n_comps``);
    * ``W_s`` — the within-batch spatial kNN graph (``k_spatial``; edge weight =
      spatial RBF x expression RBF), with Laplacian ``L_s``;

    and combines them with cross-batch anchors found by ensemble random
    projection + consensus MNN. Random projection is the natural choice for the
    anchor search because MNN is distance-based and the Johnson-Lindenstrauss
    lemma makes random projection approximately distance-preserving.

    Integration modes
    -----------------
    integration : {"laplacian", "transport", "fused"}, default "laplacian"

        ``"laplacian"`` — Laplacian-regularised embedding

            (I + lambda_anchor * L_a + lambda_spatial * L_s) Z = Z0

          ``L_a`` is the Laplacian of the ERP-consensus MNN anchor graph of the
          spatial pipeline (:func:`_build_rp_consensus_mnn_graph`:
          ``n_projections``, ``rp_dim``, ``k_mnn``, ``consensus_threshold``,
          ``mnn_strategy``; optionally reweighted by spatial context). Anchors
          act as springs between matched spots and the spatial term smooths
          each section. Defaults: ``lambda_anchor=5.0``, ``lambda_spatial=1.0``.

        ``"transport"`` — anchor corrections carried over the spatial graph

            Z = (I + lambda_spatial * L_s)^-1 (Z0 + C)

          Anchors come from the expression-only PRIME ensemble
          (:func:`_ensemble_anchor_graph`, i.e. the consensus graph of
          :func:`prime.ensemble_mnn_correct` between all batch pairs; settings
          in ``transport_anchor_kwargs``). Every anchored spot gets the
          correction ``transport_step * (weighted mean of its partners - itself)``
          in ``Z0``, and the corrections are carried to all spots of the same
          section by harmonic interpolation over the spatial graph
          (:func:`_transport_corrections`). Sections are thereby moved onto each
          other, as ``ensemble_mnn_correct`` moves cells with its correction
          vectors, but the correction varies smoothly in space rather than in
          expression. Default ``lambda_spatial=0.0`` (no smoothing afterwards).
          ``n_projections``, ``rp_dim``, ``k_mnn``, ``consensus_threshold``,
          ``mnn_strategy``, ``reweight_anchors_by_spatial_context`` and
          ``lambda_anchor`` are not used.

        ``"fused"`` — alignment and smoothing on one fused graph (experimental)

            W_f = W_s + lambda_anchor * W_a
            (L_f + eps I) C = -lambda_anchor * L_a Z0        (alignment)
            (I + lambda_spatial * L_f) Z = Z0 + C             (smoothing)

          The spatial graph (kNN without the spot itself, each edge counted
          once) and the anchor graph (``anchor_source``), both scaled to unit
          mean degree, are fused into one graph. Step 1 regresses a correction
          field on it — spatial neighbours share the correction, anchored
          partners coincide after correction, and a shift of a whole section is
          free; step 2 is the Laplacian regularisation of the ``"laplacian"``
          mode on the corrected embedding and the same graph
          (:func:`_fused_graph_regression`). ``lambda_anchor`` sets how far the
          correction may bend within a section (small: close to a rigid shift
          of each section). Defaults: ``lambda_anchor=10.0``,
          ``lambda_spatial=0.1``.

    Choosing a mode: in the ``"laplacian"`` solve the anchors only add springs
    between already similar spot pairs and the identity term keeps every other
    spot at ``Z0``, so it suits sections that differ by a moderate batch effect
    (e.g. serial sections of one tissue block). When whole sections are
    displaced by a strong technical offset (e.g. different preservation
    protocols), ``"transport"`` removes the offset by moving every spot of a
    section.

    Mode-specific parameters
    ------------------------
    lambda_anchor, lambda_spatial : float, optional
        Regularisation weights; ``None`` selects the mode default given above.
        In the ``"fused"`` mode ``lambda_anchor="auto"`` chooses the value by
        anchor cross-validation (:func:`_select_lambda_anchor_cv`); this follows
        the anchors, including their spatially coherent errors, tends to select
        large values, and is not recommended as a default.
    transport_step : float, default 0.5
        Fraction of the way each anchored spot is moved towards its partners
        (``"transport"``).
    transport_anchor_kwargs : dict, optional
        Settings of the expression-only anchor ensemble used by ``"transport"``
        and by ``"fused"`` with ``anchor_source="ensemble"``: ``n_hvg`` (2000),
        ``hvg_flavor`` ("cell_ranger"), ``n_projections`` (4), ``target_dim``
        (128), ``k_neighbors`` (15), ``consensus_threshold`` (0.4).
    anchor_source : {"ensemble", "spatial"}, default "ensemble"
        Anchor graph of the ``"fused"`` mode: the expression-only ensemble or
        the anchor graph of the ``"laplacian"`` mode. Either is reweighted by
        spatial-context similarity when
        ``reweight_anchors_by_spatial_context`` is set.
    spatial_graph_exclude_self : bool, optional
        Build the spatial kNN graph without the spot itself (see
        :func:`_build_spatial_graph`). ``None``: True for ``"fused"``, False
        otherwise.
    fused_eps : float, default 1e-4
        Ridge term of the alignment step (``"fused"``).
    fused_robust_iters : int, default 0
        Rounds of residual-based anchor reweighting in the alignment step
        (``"fused"``); 0 disables it.

    Projection-method controls
    --------------------------
    context_method : {"svd", "random_projection"}, default "svd"
        Dimensionality reduction used for the *context* embedding ``Z_gate``,
        which drives expression-distance gating of the spatial graph and
        spatial-context reweighting of anchors. ``"random_projection"`` swaps
        SVD for a distance-preserving Gaussian projection.
    base_embedding_method : {"svd", "random_projection"}, default "svd"
        Dimensionality reduction for the solver's right-hand side ``Z0``.
        Keep ``"svd"`` unless you specifically want speed over embedding
        stability: SVD/PCA ranks components by biological variance and denoises,
        whereas random projection preserves distances but does not order
        components by variance, so it changes the embedding objective.
        **Experimental.**

    Returns
    -------
    ``None`` (the embedding is written to ``adata.obsm[key_added]``) or the
    integrated copy of ``adata`` when ``copy=True``. With ``store_graphs=True``
    the anchor graph, the spatial graph, the settings and solver statistics are
    stored in ``adata.uns[graph_key]``.
    """

    if copy:
        adata = adata.copy()

    if batch_key not in adata.obs:
        raise ValueError(f"{batch_key} not in adata.obs")
    if spatial_key not in adata.obsm:
        raise ValueError(f"{spatial_key} not in adata.obsm")
    if integration not in _MODE_DEFAULTS:
        raise ValueError("integration must be 'laplacian', 'transport' or 'fused'")
    if anchor_source not in ("ensemble", "spatial"):
        raise ValueError("anchor_source must be 'ensemble' or 'spatial'")
    if context_method not in ("svd", "random_projection"):
        raise ValueError("context_method must be 'svd' or 'random_projection'")
    if base_embedding_method not in ("svd", "random_projection"):
        raise ValueError("base_embedding_method must be 'svd' or 'random_projection'")
    if isinstance(lambda_anchor, str) and not (integration == "fused" and lambda_anchor == "auto"):
        raise ValueError("lambda_anchor must be a number (or 'auto' with integration='fused')")

    if lambda_spatial is None:
        lambda_spatial = _MODE_DEFAULTS[integration]["lambda_spatial"]
    if lambda_anchor is None:
        lambda_anchor = _MODE_DEFAULTS[integration]["lambda_anchor"]
    if spatial_graph_exclude_self is None:
        spatial_graph_exclude_self = integration == "fused"

    batch_labels = adata.obs[batch_key].values
    spatial_coords = np.asarray(adata.obsm[spatial_key])
    n = adata.n_obs
    nb = len(np.unique(batch_labels))

    if verbose:
        sc.logging.info(
            f"[PRIME] n={n:,}, batches={nb}, integration={integration}, "
            f"n_hvg={n_hvg}, n_proj={n_projections}, rp_dim={rp_dim}, k_mnn={k_mnn}, "
            f"k_spatial={k_spatial}, n_comps={n_comps}"
        )

    # ---- 1) HVGs ----
    hvg_mask = _pick_hvgs(adata, batch_key=batch_key, n_hvg=n_hvg,
                          flavor=hvg_flavor, layer=layer)
    if hvg_mask.sum() < 50:
        raise RuntimeError(f"Too few HVGs selected: {int(hvg_mask.sum())}")

    # ---- 2) log1p-normalized HVG matrix (sparse) ----
    X_base = adata.layers[layer] if (layer is not None and layer in adata.layers) else adata.X
    X_log_hvg = _row_norm_log1p(X_base, target_sum=target_sum)[:, hvg_mask]

    # ---- 3) Base embeddings (Z0 and Z_gate) ----
    # Z0     : baseline biological embedding, the right-hand side of the solver.
    # Z_gate : context embedding for expression gating of the spatial graph and
    #          for spatial-context reweighting of anchors.
    # The SVD is computed once and shared whenever either embedding needs it
    # (the default for both).
    need_svd = (base_embedding_method == "svd") or (context_method == "svd")
    Z_svd = (
        _svd_embedding(X_log_hvg,
                       n_comps=max(svd_dim_for_ctx, n_comps),
                       random_state=random_state)
        if need_svd else None
    )

    # Z0 defaults to SVD/PCA: it ranks components by biological variance and
    # denoises, which is what the solver target should be. Random projection is
    # available but experimental — it preserves distances yet does not order
    # components by variance, so it changes the embedding objective.
    if base_embedding_method == "random_projection":
        Z0 = _random_projection_embedding(X_log_hvg, n_comps, random_state)
    else:
        Z0 = Z_svd[:, :n_comps].copy()

    # Z_gate may use SVD (default) or a distance-preserving random projection.
    if context_method == "random_projection":
        Z_gate = _random_projection_embedding(X_log_hvg, svd_dim_for_ctx, random_state)
    else:
        Z_gate = Z_svd[:, :svd_dim_for_ctx].copy()

    # ---- 4) Within-batch spatial graph ----
    Ws = _build_spatial_graph(
        spatial_coords, batch_labels, Z_gate,
        k_spatial=k_spatial,
        n_jobs=n_jobs,
        spatial_weight_mode="rbf",
        gate_by_expr=gate_spatial_by_expr,
        expr_gate_beta=expr_gate_beta,
        exclude_self=spatial_graph_exclude_self,
    )

    # ---- 5) Cross-batch anchor graph ----
    # "laplacian" (and "fused" with anchor_source="spatial"): ensemble random
    # projection + consensus MNN on the HVG matrix of this pipeline — the
    # prime.core principle, delegated to a reusable helper.
    # "transport" (and "fused" with anchor_source="ensemble"): the consensus
    # graph of the expression-only ensemble, all batch pairs.
    # Either way: a symmetric, self-loop-free CSR graph whose edge weights are
    # consensus frequencies.
    use_ensemble_anchors = integration == "transport" or (integration == "fused" and anchor_source == "ensemble")
    if use_ensemble_anchors:
        Wa = _ensemble_anchor_graph(
            adata, batch_labels, batch_key,
            layer=layer, target_sum=target_sum, random_state=random_state,
            **(transport_anchor_kwargs or {}),
        )
        if Wa.nnz == 0:
            raise RuntimeError(
                "No cross-batch anchors found by the expression-only ensemble. "
                "Try lowering consensus_threshold or raising k_neighbors in transport_anchor_kwargs."
            )
    else:
        Wa = _build_rp_consensus_mnn_graph(
            X_log_hvg, batch_labels,
            n_projections=n_projections,
            rp_dim=rp_dim,
            k_mnn=k_mnn,
            consensus_threshold=consensus_threshold,
            mnn_strategy=mnn_strategy,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    # Optional spatial-context reweighting: each anchor (r, c) is down-weighted
    # when the two spots sit in dissimilar spatial neighbourhoods. Not applied in
    # the "transport" mode, whose corrections are spread over space afterwards.
    if (integration != "transport" and reweight_anchors_by_spatial_context
            and Ws.nnz > 0 and Wa.nnz > 0):
        Wa = _reweight_anchors_by_spatial_context(Wa, Ws, Z_gate, spatial_power)

    if verbose:
        sc.logging.info(
            f"[PRIME] anchor edges={Wa.nnz:,}, spatial edges={Ws.nnz:,}"
        )

    graph_info: Dict[str, Any] = {}

    # ---- 6) Integration ----
    if integration == "transport":
        # Correction vectors of the anchored spots, interpolated over the spatial
        # graph; optional spatial smoothing of the corrected embedding.
        C, transport_stats = _transport_corrections(Z0, Wa, Ws, step=transport_step)
        Z1 = (Z0 + C).astype(np.float32)
        if lambda_spatial > 0:
            A = (identity(n, format="csr", dtype=np.float32)
                 + (lambda_spatial * _laplacian(Ws)).astype(np.float32))
            Z, solver_stats = _cg_solve_matrix_rhs(
                A.astype(np.float64).tocsr(), Z1,
                tol=solver_tol, maxiter=solver_maxiter,
            )
        else:
            Z, solver_stats = Z1, {}
        solver_stats = {**solver_stats, **transport_stats}
        graph_info = {
            "transport_step": float(transport_step),
            "transport_anchor_kwargs": dict(transport_anchor_kwargs or {}),
        }
        summary = f"anchored spots: {transport_stats['transport_anchored_spots']:,}"

    elif integration == "fused":
        # One fused graph (unit-mean-degree spatial + anchor graphs): Laplacian
        # regression of the correction field, then smoothing on the same graph.
        Wa_u, deg_a = _unit_mean_degree(Wa)
        Ws_u, deg_s = _unit_mean_degree(Ws)
        cv_scores: Dict[str, float] = {}
        if isinstance(lambda_anchor, str):      # "auto"
            lambda_anchor, cv_scores = _select_lambda_anchor_cv(
                Z0, Wa_u, Ws_u, eps=fused_eps, random_state=random_state,
            )
        Z, solver_stats = _fused_graph_regression(
            Z0, Wa_u, Ws_u, batch_labels,
            lambda_anchor=lambda_anchor,
            lambda_spatial=lambda_spatial,
            eps=fused_eps,
            robust_iters=fused_robust_iters,
            smooth_tol=solver_tol,
            smooth_maxiter=solver_maxiter,
        )
        solver_stats.update({f"cv_resid_{k}": v for k, v in cv_scores.items()})
        solver_stats.update(
            anchor_edges=int(Wa.nnz),
            spatial_edges=int(Ws.nnz),
            anchored_spots=int((np.diff(Wa.indptr) > 0).sum()),
            anchor_mean_degree=deg_a,
            spatial_mean_degree=deg_s,
        )
        graph_info = {
            "anchor_source": anchor_source,
            "fused_eps": float(fused_eps),
            "fused_robust_iters": int(fused_robust_iters),
            "transport_anchor_kwargs": dict(transport_anchor_kwargs or {}),
            "spatial_graph_exclude_self": bool(spatial_graph_exclude_self),
        }
        summary = (f"anchored spots: {solver_stats['anchored_spots']:,}, section-shift share of the "
                   f"correction: {solver_stats['correction_section_shift_share']:.2f}")

    else:
        # Laplacian-regularized CG solve.
        La = _laplacian(Wa)
        Ls = _laplacian(Ws)

        A = (identity(n, format="csr", dtype=np.float32)
             + (lambda_anchor * La).astype(np.float32)
             + (lambda_spatial * Ls).astype(np.float32))
        A.eliminate_zeros()

        Z, solver_stats = _cg_solve_matrix_rhs(
            A.astype(np.float64).tocsr(),
            Z0.astype(np.float32),
            tol=solver_tol,
            maxiter=solver_maxiter,
        )
        summary = f"cg converged dims: {solver_stats['cg_converged_dims']}/{n_comps}"

    adata.obsm[key_added] = Z.astype(np.float32)

    if store_graphs:
        adata.uns[graph_key] = {
            "W_anchor": Wa,
            "W_spatial": Ws,
            "integration": integration,
            "n_hvg": int(hvg_mask.sum()),
            "anchor_projection_method": "random_projection",
            "context_method": context_method,
            "base_embedding_method": base_embedding_method,
            "n_projections": int(n_projections),
            "rp_dim": int(rp_dim),
            "svd_dim_for_ctx": int(svd_dim_for_ctx),
            "k_mnn": int(k_mnn),
            "consensus_threshold": float(consensus_threshold),
            "mnn_strategy": mnn_strategy,
            "k_spatial": int(k_spatial),
            "random_state": int(random_state),
            "lambda_anchor": float(lambda_anchor),
            "lambda_spatial": float(lambda_spatial),
            "reweight_anchors_by_spatial_context": bool(reweight_anchors_by_spatial_context),
            "spatial_power": float(spatial_power),
            "gate_spatial_by_expr": bool(gate_spatial_by_expr),
            "expr_gate_beta": float(expr_gate_beta),
            "solver": "cg",
            **graph_info,
            **{k: v for k, v in solver_stats.items()
               if integration == "laplacian" or k != "cg_info_per_dim"},
        }

    if verbose:
        sc.logging.info(
            f"[PRIME] stored adata.obsm['{key_added}'] ({summary})"
        )

    return adata if copy else None
