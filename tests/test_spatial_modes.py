"""Tests for the integration modes of prime.prime_st ("laplacian", "transport", "fused").

Run with:  pytest tests/test_spatial_modes.py
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from prime import prime_st
from prime.spatial import _build_spatial_graph

anndata = pytest.importorskip("anndata")


def _make_synthetic(n_per_batch=150, n_genes=300, n_types=4, n_batches=3, seed=0):
    """Cell type follows the spatial quadrant; batches differ by a per-gene multiplicative effect."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    programs = rng.gamma(1.0, 1.0, size=(n_types, n_genes)) + 0.05
    Xs, batches, coords, types = [], [], [], []
    for b in range(n_batches):
        xy = rng.uniform(0.0, 1.0, size=(n_per_batch, 2))
        t = ((xy[:, 0] > 0.5).astype(int) + 2 * (xy[:, 1] > 0.5).astype(int)) % n_types
        factor = np.exp(rng.normal(0.0, 0.5, size=n_genes))
        Xs.append(rng.poisson(programs[t] * factor[None, :] * 3.0).astype(np.float32))
        batches += [f"batch{b}"] * n_per_batch
        coords.append(xy)
        types += list(t)
    adata = anndata.AnnData(
        X=csr_matrix(np.vstack(Xs)),
        obs=pd.DataFrame({"batch": pd.Categorical(batches), "cell_type": pd.Categorical(np.array(types).astype(str))}),
    )
    adata.obsm["spatial"] = np.vstack(coords).astype(np.float32)
    return adata


BASE = dict(batch_key="batch", n_hvg=200, n_comps=20, random_state=0, verbose=False,
            transport_anchor_kwargs={"n_hvg": 200})


def _run(adata, **kw):
    a = adata.copy()
    prime_st(a, store_graphs=True, **{**BASE, **kw})
    return a


@pytest.fixture(scope="module")
def adata():
    return _make_synthetic()


@pytest.mark.parametrize("mode", ["laplacian", "transport", "fused"])
def test_modes_run_and_are_deterministic(adata, mode):
    a, b = _run(adata, integration=mode), _run(adata, integration=mode)
    Z = a.obsm["X_prime"]
    assert Z.shape == (adata.n_obs, 20) and Z.dtype == np.float32 and np.isfinite(Z).all()
    assert np.array_equal(Z, b.obsm["X_prime"])
    g = a.uns["prime_graphs"]
    assert g["integration"] == mode
    for key in ("W_anchor", "W_spatial"):
        W = g[key]
        assert abs(W - W.T).max() < 1e-6 and W.diagonal().sum() == 0      # symmetric, no self-loops


def test_default_mode_is_laplacian_with_its_default_weights(adata):
    default = _run(adata)
    explicit = _run(adata, integration="laplacian", lambda_anchor=5.0, lambda_spatial=1.0)
    assert np.array_equal(default.obsm["X_prime"], explicit.obsm["X_prime"])
    g = default.uns["prime_graphs"]
    assert (g["lambda_anchor"], g["lambda_spatial"]) == (5.0, 1.0)


def test_mode_defaults_of_the_weights(adata):
    assert _run(adata, integration="transport").uns["prime_graphs"]["lambda_spatial"] == 0.0
    g = _run(adata, integration="fused").uns["prime_graphs"]
    assert (g["lambda_anchor"], g["lambda_spatial"]) == (10.0, pytest.approx(0.1))
    assert g["spatial_graph_exclude_self"] is True


def test_zero_weights_return_the_base_embedding(adata):
    base = _run(adata, lambda_anchor=0.0, lambda_spatial=0.0).obsm["X_prime"]
    lap = _run(adata).obsm["X_prime"]
    assert not np.allclose(base, lap)
    # transport with step 0 and no smoothing leaves the base embedding untouched
    still = _run(adata, integration="transport", transport_step=0.0).obsm["X_prime"]
    np.testing.assert_allclose(still, base, atol=1e-5)


@pytest.mark.parametrize("mode", ["transport", "fused"])
def test_alignment_modes_reduce_the_section_offset(adata, mode):
    """The distance between batch centroids of the same cell type shrinks relative to the base embedding."""
    batch, types = adata.obs["batch"].values, adata.obs["cell_type"].values

    def offset(Z):
        d = []
        for t in np.unique(types):
            cents = [Z[(types == t) & (batch == b)].mean(0) for b in np.unique(batch)]
            d += [np.linalg.norm(cents[i] - cents[j]) for i in range(len(cents)) for j in range(i + 1, len(cents))]
        return np.mean(d)

    base = _run(adata, lambda_anchor=0.0, lambda_spatial=0.0).obsm["X_prime"]
    assert offset(_run(adata, integration=mode).obsm["X_prime"]) < offset(base)


def test_fused_options(adata):
    for kw in (dict(anchor_source="spatial"), dict(fused_robust_iters=1), dict(lambda_anchor="auto")):
        a = _run(adata, integration="fused", **kw)
        assert np.isfinite(a.obsm["X_prime"]).all()
    assert isinstance(a.uns["prime_graphs"]["lambda_anchor"], float)       # "auto" resolved to a number


def test_spatial_graph_without_self_has_k_neighbours():
    rng = np.random.default_rng(0)
    xy = rng.uniform(size=(80, 2))
    batch = np.array(["a"] * 80)
    Z = rng.normal(size=(80, 5)).astype(np.float32)
    k = 6
    with_self = _build_spatial_graph(xy, batch, Z, k_spatial=k, exclude_self=False)
    without = _build_spatial_graph(xy, batch, Z, k_spatial=k, exclude_self=True)
    # every spot keeps at least k (k - 1 with the self-neighbour convention) neighbours after symmetrisation
    assert np.diff(without.indptr).min() >= k
    assert np.diff(with_self.indptr).min() >= k - 1
    assert without.diagonal().sum() == 0 and abs(without - without.T).max() < 1e-7


def test_invalid_arguments(adata):
    with pytest.raises(ValueError):
        _run(adata, integration="other")
    with pytest.raises(ValueError):
        _run(adata, integration="fused", anchor_source="other")
    with pytest.raises(ValueError):
        _run(adata, integration="laplacian", lambda_anchor="auto")
