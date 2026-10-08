"""Tests for prime.core: consensus graph and ensemble_mnn_correct.

Run with:  pytest tests/test_core.py
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from prime import build_consensus_graph, ensemble_mnn_correct
from prime.core import _find_mutual_neighbors_sets, _self_knn

anndata = pytest.importorskip("anndata")


def _make_batches(n_per_batch=150, n_genes=200, n_types=3, n_batches=3, seed=0, sparse=True):
    """Shared cell types across batches with a per-gene batch shift (log-scale expression)."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    programs = rng.gamma(1.0, 1.0, size=(n_types, n_genes)) + 0.05
    Xs, batches, types = [], [], []
    for b in range(n_batches):
        t = rng.integers(0, n_types, size=n_per_batch)
        factor = np.exp(rng.normal(0.0, 0.3, size=n_genes))
        Xs.append(np.log1p(rng.poisson(programs[t] * factor[None, :] * 3.0)).astype(np.float32))
        batches += [f"batch{b}"] * n_per_batch
        types += list(t)
    X = np.vstack(Xs)
    return anndata.AnnData(
        X=csr_matrix(X) if sparse else X,
        obs=pd.DataFrame({"batch": pd.Categorical(batches), "cell_type": pd.Categorical(np.array(types).astype(str))}),
    )


KW = dict(n_projections=4, target_dim=32, k_neighbors=10, consensus_threshold=0.4, sigma=0.1, random_state=0)


def test_mutual_neighbors_match_set_definition():
    rng = np.random.default_rng(0)
    a_to_b = np.array([rng.choice(30, size=5, replace=False) for _ in range(40)])
    b_to_a = np.array([rng.choice(40, size=5, replace=False) for _ in range(30)])
    forward = {(i, j) for i in range(40) for j in a_to_b[i]}
    expected = [(i, j) for j in range(30) for i in b_to_a[j] if (i, j) in forward]
    got = _find_mutual_neighbors_sets(a_to_b, b_to_a)
    assert got.shape == (len(expected), 2)
    assert [tuple(p) for p in got.tolist()] == expected


def test_self_knn_matches_sklearn():
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 20)).astype(np.float32)
    dist, ind = _self_knn(X, 10)
    ref_dist, ref_ind = NearestNeighbors(n_neighbors=10, algorithm="brute").fit(X).kneighbors(X)
    assert np.array_equal(ind[:, 0], np.arange(300))                 # every row is its own nearest neighbour
    assert np.mean(ind == ref_ind) > 0.99                            # ties aside, the same neighbours
    np.testing.assert_allclose(dist, ref_dist, atol=1e-4)


def test_consensus_graph_is_cross_batch_and_thresholded():
    adata = _make_batches()
    batch = adata.obs["batch"].values
    G = build_consensus_graph(adata.X, batch, None, 5, 32, 10, 0.4, 0)
    assert G.shape == (adata.n_obs, adata.n_obs) and G.nnz > 0
    r, c = G.nonzero()
    assert np.all(batch[r] != batch[c])                              # MNN edges only join different batches
    assert G.data.min() >= 0.4 - 1e-6 and G.data.max() <= 1.0 + 1e-6    # weights = vote fractions above threshold
    assert abs(G - G.T).max() < 1e-7                                 # every edge is stored in both directions
    strict = build_consensus_graph(adata.X, batch, None, 5, 32, 10, 1.0, 0)
    assert strict.nnz <= G.nnz


@pytest.mark.parametrize("sparse", [True, False])
def test_correction_shape_determinism_and_chunking(sparse):
    adata = _make_batches(sparse=sparse)
    Z1 = ensemble_mnn_correct(adata.copy(), "batch", inplace=False, **KW)
    Z2 = ensemble_mnn_correct(adata.copy(), "batch", inplace=False, **KW)
    Z3 = ensemble_mnn_correct(adata.copy(), "batch", inplace=False, chunk_size=37, **KW)
    assert isinstance(Z1, np.ndarray) and Z1.shape == adata.shape
    assert np.array_equal(Z1, Z2)                                    # deterministic for a fixed seed
    np.testing.assert_allclose(Z1, Z3, rtol=1e-5, atol=1e-5)         # gene chunking does not change the result
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    assert not np.allclose(Z1, X)                                    # a correction was applied


def test_correction_brings_batches_together():
    adata = _make_batches()
    X = adata.X.toarray()
    Z = ensemble_mnn_correct(adata.copy(), "batch", inplace=False, **KW)
    batch, types = adata.obs["batch"].values, adata.obs["cell_type"].values

    def batch_gap(M):                                                # distance between batch centroids, per cell type
        gaps = []
        for t in np.unique(types):
            cents = [M[(types == t) & (batch == b)].mean(0) for b in np.unique(batch)]
            gaps += [np.linalg.norm(cents[i] - cents[j]) for i in range(len(cents)) for j in range(i + 1, len(cents))]
        return np.mean(gaps)

    assert batch_gap(Z) < batch_gap(X)


def test_no_anchor_case_returns_the_input_matrix():
    adata = _make_batches()
    Z = ensemble_mnn_correct(adata.copy(), "batch", inplace=False, **{**KW, "consensus_threshold": 2.0})
    assert isinstance(Z, np.ndarray)
    assert np.array_equal(Z, adata.X.toarray())
