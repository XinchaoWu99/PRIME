# PRIME

**P**rojection-based **R**obust **I**ntegration via **M**anifold **E**mbedding

[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PRIME is a Python package for **batch effect correction** in single-cell RNA-seq (scRNA-seq) and **multi-slice integration** in spatial transcriptomics. It combines:

- **Ensemble random projections** of the expression matrix
- **Consensus mutual nearest neighbor (MNN) graphs** built across many low-dimensional views
- **(Spatial only)** Laplacian-regularized embedding that jointly respects MNN anchors and within-slice spatial neighborhoods, with an anchor-transport mode for sections separated by a strong technical offset

Two main entry points cover both modalities:

| Function | Use case | Output |
|----------|----------|--------|
| `prime.ensemble_mnn_correct` | scRNA-seq batch correction | Corrected expression matrix (`n_cells × n_genes`) |
| `prime.prime_st` | Spatial transcriptomics integration | Integrated embedding in `adata.obsm["X_prime"]` |

---

## Installation

### Option A — conda / mamba (recommended)

The repository ships ready-to-use environment files. From a clone of the repo:

```bash
git clone https://github.com/XinchaoWu99/PRIME.git
cd PRIME

# CPU / base environment (Linux, macOS, Windows)
conda env create -f environment.yml      # or: mamba env create -f environment.yml
conda activate prime
```

`environment.yml` pulls the full scientific stack from conda-forge and then
installs PRIME itself in editable mode, so `import prime` works immediately.
Run the command **from the repository root** (so the `-e .` step resolves to
this repo). [`mamba`](https://mamba.readthedocs.io/) is a faster drop-in
replacement for `conda` and is recommended — especially for the GPU
environment below.

### Option B — pip

```bash
git clone https://github.com/XinchaoWu99/PRIME.git
cd PRIME
pip install -e .                 # core
pip install -e ".[all]"          # + plotting & metrics extras
```

**Requirements:** Python ≥ 3.9, NumPy, Pandas, SciPy, scikit-learn, AnnData, Scanpy.

### GPU backend (optional, NVIDIA only)

The VRAM-bounded GPU path (`prime.gpu.ensemble_mnn_correct`) additionally needs
cupy and a GPU ANN backend (cuvs or faiss-gpu), and a **Linux machine with an
NVIDIA GPU**. The simplest route is the dedicated environment file — edit its
`cuda-version` to match your driver (`nvidia-smi`) first:

```bash
mamba env create -f environment-gpu.yml
conda activate prime-gpu
python -c "import prime.gpu; print(prime.gpu.detect())"
```

Or layer it onto an existing install with pip, picking a cupy wheel that matches
your CUDA version:

```bash
pip install cupy-cuda12x
pip install "prime-sc[gpu-cuvs]"     # recommended
# or
pip install "prime-sc[gpu-faiss]"
```

See [Scaling to 1M+ cells](#scaling-to-1m-cells) for GPU usage.

---

## Quick start

### 1. scRNA-seq batch correction

```python
import scanpy as sc
import prime

# adata.X: cell × gene matrix (raw counts or normalized expression)
# adata.obs["batch"]: batch labels per cell
adata = sc.read_h5ad("your_data.h5ad")

# Run PRIME — returns the batch-corrected expression matrix
X_corrected = prime.ensemble_mnn_correct(
    adata,
    batch_key="batch",
    n_projections=10,        # number of random projections
    target_dim=50,           # dimension of each random projection
    k_neighbors=20,          # k for MNN search
    consensus_threshold=0.4, # fraction of projections an edge must appear in
    sigma=0.1,               # Gaussian smoothing bandwidth
    random_state=42,
)

# Store and use downstream
adata.layers["prime"] = X_corrected
adata.X = X_corrected

sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["batch", "cell_type"])
```

### 2. Spatial transcriptomics integration

For Visium / Stereo-seq / MERFISH and similar datasets with multiple slices:

```python
import scanpy as sc
import prime

# adata.obs["slice"]: slice / batch ID
# adata.obsm["spatial"]: (n_cells, 2) array of spatial coordinates
adata = sc.read_h5ad("multi_slice_spatial.h5ad")

prime.prime_st(
    adata,
    batch_key="slice",
    spatial_key="spatial",
    n_hvg=3000,
    n_projections=10,
    k_mnn=20,
    k_spatial=6,
    lambda_anchor=5.0,     # weight of cross-slice MNN constraint
    lambda_spatial=1.0,    # weight of within-slice spatial smoothness
    n_comps=30,
    key_added="X_prime",
)

# Integrated embedding is now in adata.obsm["X_prime"]
sc.pp.neighbors(adata, use_rep="X_prime")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["slice", "region"])
```

---

## API reference

### `prime.ensemble_mnn_correct`

```python
prime.ensemble_mnn_correct(
    adata,
    batch_key,
    projection_keys=None,
    n_projections=10,
    target_dim=50,
    k_neighbors=20,
    consensus_threshold=0.4,
    sigma=0.1,
    random_state=42,
    key_added=None,
    inplace=True,
    chunk_size=2000,
)
```

**Key parameters**

| Parameter | Description |
|-----------|-------------|
| `adata` | `AnnData` with expression matrix in `.X` and batch labels in `.obs[batch_key]`. |
| `batch_key` | Column name in `adata.obs` holding batch IDs. |
| `n_projections` | Number of random projections in the ensemble. More projections → more robust consensus, higher cost. |
| `target_dim` | Output dimension of each random projection. |
| `k_neighbors` | Number of nearest neighbors per cell when building MNN graphs. |
| `consensus_threshold` | An MNN edge is kept only if it appears in ≥ this fraction of projections (0 – 1). |
| `sigma` | Bandwidth of the Gaussian smoothing kernel applied after MNN correction. |
| `chunk_size` | Number of genes processed per chunk; lower → less memory. |

**Returns:** `np.ndarray` of shape `(n_cells, n_genes)` containing the batch-corrected expression matrix. Assign it back to `adata.X` or `adata.layers[...]`.

### `prime.prime_st`

```python
prime.prime_st(
    adata,
    *,
    batch_key,
    spatial_key="spatial",
    layer=None,
    n_hvg=3000,
    hvg_flavor="seurat_v3",
    integration="laplacian",      # "laplacian", "transport" or "fused"
    n_projections=10,
    rp_dim=50,
    k_mnn=20,
    consensus_threshold=0.4,
    mnn_strategy="star",          # "star" or "pairwise"
    k_spatial=6,
    gate_spatial_by_expr=True,
    reweight_anchors_by_spatial_context=True,
    n_comps=30,
    lambda_anchor=None,           # None: default of the mode (laplacian 5.0)
    lambda_spatial=None,          # None: default of the mode (laplacian 1.0, transport 0.0)
    solver_tol=1e-5,
    solver_maxiter=200,
    transport_step=0.5,
    transport_anchor_kwargs=None,
    random_state=0,
    key_added="X_prime",
    store_graphs=False,
    copy=False,
    verbose=True,
)
```

**Key parameters**

| Parameter | Description |
|-----------|-------------|
| `adata` | `AnnData` with counts in `.X`, batch labels in `.obs[batch_key]`, spatial coordinates in `.obsm[spatial_key]`. |
| `n_hvg` | Number of highly variable genes used as the SVD input. |
| `k_mnn` | k for cross-slice MNN search. |
| `k_spatial` | k for within-slice spatial kNN graph. |
| `mnn_strategy` | `"star"` anchors all slices to a hub, `"pairwise"` builds all pairwise MNNs (slower, denser). |
| `integration` | How anchors and the spatial graph are combined: `"laplacian"` (default), `"transport"` or `"fused"` (below). |
| `lambda_anchor` | Weight on the cross-slice MNN Laplacian regularizer (`None`: default of the mode). |
| `lambda_spatial` | Weight on the within-slice spatial Laplacian regularizer (`None`: default of the mode). |
| `n_comps` | Output embedding dimension. |
| `key_added` | Slot in `adata.obsm` where the integrated embedding is stored. |
| `copy` | If `True`, return a corrected copy of `adata` instead of modifying in place. |

**Returns:** `None` (writes `adata.obsm[key_added]`) or a new `AnnData` if `copy=True`.

The integration solves

```
(I + λ_anchor · L_anchor + λ_spatial · L_spatial) · Z = Z₀
```

where `L_anchor` is the Laplacian of the consensus MNN anchor graph, `L_spatial` is the within-slice spatial Laplacian, and `Z₀` is the TruncatedSVD embedding of HVG log1p-normalized expression. The linear system is solved column-wise by conjugate gradient.

**Integration modes.** The equation above is the default mode, `integration="laplacian"`: anchors act as springs between matched spots and the spatial term smooths each slice. It suits slices that differ by a moderate batch effect, such as serial sections of one tissue block.

| `integration` | What it does | Use it when |
|---------------|--------------|-------------|
| `"laplacian"` (default) | Solves `(I + λ_anchor·L_anchor + λ_spatial·L_spatial)·Z = Z₀`. | Slices are already roughly comparable (e.g. serial sections). |
| `"transport"` | Finds anchors with the expression-only PRIME ensemble (the consensus graph of `ensemble_mnn_correct`, all slice pairs), moves every anchored spot `transport_step` of the way to its partners, and carries these corrections to all spots of the slice by harmonic interpolation over the spatial graph: `Z = (I + λ_spatial·L_spatial)⁻¹ (Z₀ + C)`. | Whole slices are displaced by a strong technical offset (e.g. FFPE vs. fresh-frozen vs. fixed-frozen preparations). |
| `"fused"` (experimental) | Fuses the spatial graph and the anchor graph into one graph, regresses a correction field on it, then smooths on the same graph. | Exploring a single-graph formulation of the two modes above. |

```python
# Sections prepared with different protocols: align them by transporting anchor corrections over space
prime.prime_st(adata, batch_key="sample", integration="transport")
```

`transport_anchor_kwargs` passes the settings of the anchor ensemble used by `"transport"` (`n_hvg`, `n_projections`, `target_dim`, `k_neighbors`, `consensus_threshold`). See the docstring of `prime.prime_st` for all mode-specific parameters.

---

## Scaling to 1M+ cells

PRIME ships two optional subpackages that lift the usual RAM and VRAM bottlenecks at large scale.

### `prime.pp` — streaming preprocessing (no full matrix in RAM)

The most common reason scanpy preprocessing blows up on 1M+ cells is the HVG step: `sc.pp.highly_variable_genes(flavor="seurat_v3")` can silently densify intermediates. `prime.pp.streaming_hvg` solves this with two streaming passes over backed-mode AnnData, using only `O(n_genes × n_batches)` extra memory regardless of the cell count.

```python
import prime

# Works on either an in-memory AnnData or a path to an .h5ad file.
hvg_mask = prime.pp.streaming_hvg(
    "huge_dataset.h5ad",   # backed-mode read; peak memory ~ 1 chunk worth
    batch_key="batch",
    n_top_genes=3000,
    chunk_size=20_000,
    min_cells=10,
    aggregation="median_rank",   # Seurat-v3-style cross-batch rank aggregation
)
# hvg_mask is a bool array of shape (n_genes,)
```

For a quick "drop the sparsely-expressed genes" pass that doesn't even need normalization:

```python
keep = prime.pp.streaming_gene_filter("huge_dataset.h5ad", min_cells=10)
```

### `prime.gpu.ensemble_mnn_correct` — GPU backend, VRAM-bounded

The GPU version of `ensemble_mnn_correct` integrates the techniques needed to keep 1M cells well under 16 GB of VRAM:

| Technique | Implementation |
|-----------|----------------|
| Never densify X | Sparse CSR on GPU via cuSPARSE matmul |
| Approximate kNN, batched queries | cuvs CAGRA (or faiss-gpu), `knn_chunk_size` rows per call |
| Mixed precision | `projection_dtype="float16"` for the projection / kNN intermediate |
| Free pool between projections | Explicit `cp.get_default_memory_pool().free_all_blocks()` each iteration |
| CPU correction step | Gene-chunked additive update — already RAM-bounded by `chunk_size` |

```python
import prime
import prime.gpu

# Inspect which backends are available before running:
print(prime.gpu.detect())
# GPUEnv(cupy=True, cupy_sparse=True, cuvs=True, faiss_gpu=False)

X_corrected = prime.gpu.ensemble_mnn_correct(
    adata,
    batch_key="batch",
    n_projections=10,
    target_dim=50,
    k_neighbors=20,
    consensus_threshold=0.4,
    knn_chunk_size=50_000,        # cap query VRAM
    knn_backend="auto",           # "cuvs" preferred, falls back to "faiss"
    projection_dtype="float16",   # half the projection VRAM
    chunk_size=2000,              # CPU correction chunking
)
adata.layers["prime_gpu"] = X_corrected
```

**Installation.** Use the dedicated GPU environment (`environment-gpu.yml`, Linux + NVIDIA only) — see [Installation → GPU backend](#gpu-backend-optional-nvidia-only). Or layer it onto an existing environment with a cupy wheel matching your CUDA version plus one ANN backend:

```bash
pip install cupy-cuda12x
pip install "prime-sc[gpu-cuvs]"     # recommended
# or
pip install "prime-sc[gpu-faiss]"
```

### Typical end-to-end pipeline for 1M cells

```python
import scanpy as sc
import prime

# 1) Streaming HVG straight from disk — no full matrix loaded.
hvg = prime.pp.streaming_hvg("data.h5ad", batch_key="batch", n_top_genes=3000)

# 2) Load only the HVG slice into RAM.
adata = sc.read_h5ad("data.h5ad", backed="r")
adata = adata[:, hvg].to_memory()

# 3) Run PRIME on GPU.
X_corrected = prime.gpu.ensemble_mnn_correct(adata, batch_key="batch")
adata.X = X_corrected

# 4) Downstream as usual.
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
```

---

## Evaluation metrics

PRIME ships with two evaluation utilities useful for benchmarking integration quality.

### Cross-slice Layer Continuity (XLC)

For data with ordinal labels (e.g., cortical layers L1–L6/WM), measures whether neighborhood structure preserves the layer ordering:

```python
from prime.metrics import xlc_score

scores = xlc_score(
    adata,
    label_key="layer",
    embedding_keys=["X_prime", "X_pca", "X_harmony"],
    batch_key="slice",
    k_values=(15, 30, 50),
    cross_slice=True,
    n_perm=100,
)
print(scores)   # DataFrame with one row per embedding
```

### Isolated label preservation

Silhouette-based score for cell types that appear in only a few batches — a stress test for over-correction:

```python
from prime.metrics import compute_isolated_label_scores

iso_scores = compute_isolated_label_scores(
    adata,
    label_key="cell_type",
    embedding_keys=["X_prime", "X_pca", "X_harmony"],
    batch_key="batch",
)
print(iso_scores)
```

---

## Visualization

Render scIB-style benchmark result tables with colored cells and per-metric bar plots:

```python
from prime.plotting import plot_scib_results_table, save_scib_results_publication_pdf

fig, ax, tab = plot_scib_results_table(results_df)
save_scib_results_publication_pdf(fig, ax, "benchmark_table.pdf")
```

Requires the optional `plotting` extras: `pip install -e ".[plotting]"`.

---

## Module layout

```
prime/
├── core.py         # ensemble_mnn_correct (scRNA-seq, CPU)
├── spatial.py      # prime_st (spatial transcriptomics, CPU)
├── pp/
│   └── streaming.py      # streaming_hvg, streaming_gene_filter
├── gpu/
│   ├── ensemble.py       # GPU ensemble_mnn_correct (VRAM-bounded)
│   ├── _knn.py           # chunked cuvs / faiss-gpu ANN wrapper
│   └── _backend.py       # lazy GPU backend detection
├── metrics/
│   ├── xlc.py            # ordinal_layer_continuity, xlc_score
│   └── isolated_label.py # compute_isolated_label_scores, ...
└── plotting/
    └── benchmark.py      # plot_scib_results_table, ...
```

---

## Reproducing the analyses of the paper

Besides the package, the repository holds the code behind the results of the paper:

```
pipeline/     analysis scripts: integration benchmarks, parameter / anchor / ablation analyses of PRIME, spatial
              benchmarks (DLPFC, Visium HD mouse brain), scaling, trajectories, scGPT zero-shot comparison
notebooks/    one figure notebook per data set, drawing from the saved results of the pipeline
tests/        unit tests of the package (pytest)
```

1. Set the data and output locations in [`pipeline/config.yaml`](pipeline/config.yaml) (or in an untracked
   `pipeline/config.local.yaml`).
2. Run the analyses you need. Every script of `pipeline/` states its command line, inputs and outputs in its header;
   the header of [`pipeline/common.py`](pipeline/common.py) maps the folders.
3. Draw the figures:

| Notebook | Data | Figures |
|----------|------|---------|
| [`01_human_immune`](notebooks/01_human_immune.ipynb) | Human immune (33,506 cells, 10 batches) | scIB table, UMAPs, markers, Monocle3 trajectories, parameter sensitivity, trajectory comparison, scGPT |
| [`02_NSCLC_lung_adenocarcinoma`](notebooks/02_NSCLC_lung_adenocarcinoma.ipynb) | NSCLC atlas (410,927 / 892,296 cells) | scIB table, UMAPs, robustness to the number of datasets, scaling, scGPT |
| [`03_DLPFC`](notebooks/03_DLPFC.ipynb) | DLPFC, 12 Visium sections | spatial benchmark table, UMAPs, λ sensitivity, setting selection, spatial domains and markers |
| [`04_Jurkat_293T`](notebooks/04_Jurkat_293T.ipynb) | Jurkat / 293T mixture | anchor validation, module ablation |
| [`05_VisiumHD_mouse_brain`](notebooks/05_VisiumHD_mouse_brain.ipynb) | Visium HD mouse brain, three preservation protocols | integration of the three sections, region / layer labels |

Run a notebook interactively, or all of them on a SLURM cluster with `bash notebooks/run_all.sh`. Figures are
written to `figures/` (not tracked).

---

## Citation

If you use PRIME in your research, please cite (manuscript in preparation):

> Wu, X. *et al.* **PRIME: Projection-based Robust Integration via Manifold Embedding for single-cell and spatial transcriptomics.** *In preparation* (2026).

A BibTeX entry will be added here once the manuscript is published.

---

## License

PRIME is released under the [MIT License](LICENSE).
