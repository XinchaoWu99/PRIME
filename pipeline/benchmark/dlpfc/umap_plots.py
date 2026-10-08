"""DLPFC (12 Visium sections): UMAP of every integration method, coloured by section and by layer.

Cell-structured script (`# %%` cells). Same inputs as benchmark_metrics.py; point-only PNGs are written to
<benchmark.dlpfc.results_dir>/umap_plots/.
"""
# %%
import os
import sys
from pathlib import Path

import scanpy as sc
import anndata as ad

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # pipeline/ (common.py)
import common as C  # noqa: E402

CFG = C.load_config()
# %%
DLFPC_data_dir = CFG["benchmark"]["dlpfc"]["data_dir"]
DLFPC_merged_data_file = os.path.join(DLFPC_data_dir, "DLPFC_merged.h5ad")

DLFPC_adata = sc.read_h5ad(DLFPC_merged_data_file)
DLFPC_adata

adata = DLFPC_adata.copy()
adata.obs['section'] = adata.obs['sample'].copy()

# %% run pca
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3')
adata_tmp = adata[:, adata.var['highly_variable']].copy()
sc.pp.scale(adata_tmp, max_value=10)
sc.tl.pca(adata_tmp, n_comps=50, svd_solver='arpack')
adata.obsm['Unintegrated'] = adata_tmp.obsm['X_pca']

# %% loading precast embedding
save_dir = CFG["benchmark"]["dlpfc"]["results_dir"]
precast_df = pd.read_csv(f"{save_dir}/PRECAST_embedding_with_meta.csv")

precast_df['key'] = precast_df['section'].astype(str) + '_' + precast_df['barcode'].astype(str)
embedding_cols = [f"PRECAST_{i}" for i in range(1, 16)]
precast_df = precast_df.set_index('key')
all_keys = adata.obs['key'].values
missing_keys = set(all_keys) - set(precast_df.index)
mean_embedding = precast_df[embedding_cols].mean(axis=0).values  # shape: (15,)
for mk in missing_keys:
    precast_df.loc[mk, embedding_cols] = mean_embedding
    section_val = adata.obs.loc[adata.obs['key'] == mk, 'section'].iloc[0]
    barcode_val = adata.obs.loc[adata.obs['key'] == mk, 'barcode'].iloc[0]
    precast_df.loc[mk, 'section'] = section_val
    precast_df.loc[mk, 'barcode'] = barcode_val

embedding_matrix = precast_df.loc[all_keys, embedding_cols].values  # shape: (N, 15)
adata.obsm['X_precast'] = embedding_matrix

print(f"adata.obsm['X_precast'] shape: {adata.obsm['X_precast'].shape}")
# %% loading embeddings
results_dir = CFG["benchmark"]["dlpfc"]["results_dir"]
methods_list = [
    "STAligner", "paste", "GraphST", "prime"
]

for method in methods_list:
    embedding_file = os.path.join(results_dir, f"{method}_embedding.npy")
    if os.path.exists(embedding_file):
        embedding = np.load(embedding_file)
        adata.obsm[f"X_{method}"] = embedding
        print(f"Loaded embedding for {method} from {embedding_file}")
    else:
        print(f"Embedding file for {method} not found at {embedding_file}")

benchmark_keys = ["STAligner", "paste", "GraphST", "prime", "precast"]
benchmark_keys = [f"X_{method}" for method in benchmark_keys] + ["Unintegrated"]

# %%
from pathlib import Path

PNG_DPI = 300
DEFAULT_UMAP_FIGSIZE = (4, 4)
DEFAULT_UMAP_COLORS = ("section", "layer_guess")

def _embedding_plot_kwargs(
    *,
    color: str,
    point_size: float | None,
    palette,
    cmap,
    alpha: float,
    ax,
    frameon: bool,
    legend_loc,
    colorbar_loc,
    title,
) -> dict:
    kwargs = {
        "color": color,
        "sort_order": True,
        "frameon": frameon,
        "legend_loc": legend_loc,
        "colorbar_loc": colorbar_loc,
        "title": title,
        "show": False,
        "ax": ax,
        "alpha": alpha,
        "size": point_size,
    }
    if palette is not None:
        kwargs["palette"] = palette
    if cmap is not None:
        kwargs["color_map"] = cmap
        kwargs["cmap"] = cmap
    return kwargs
 

def plot_points_only_png(
    adata,
    *,
    basis_key: str,
    color: str,
    output_path: Path,
    figsize: tuple[float, float],
    png_dpi: int,
    point_size: float | None,
    palette,
    cmap,
    alpha: float,
    transparent: bool,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.embedding(
        adata,
        basis=basis_key,
        **_embedding_plot_kwargs(
            color=color,
            point_size=point_size,
            palette=palette,
            cmap=cmap,
            alpha=alpha,
            ax=ax,
            frameon=False,
            legend_loc=None,
            colorbar_loc=None,
            title=None,
        ),
    )
    ax.set_axis_off()
    fig.savefig(
        output_path,
        dpi=png_dpi,
        transparent=transparent,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def _umap_output_path(*, data_path: str | Path, method_slug: str, color: str) -> Path:
    return Path(data_path) / f"{method_slug}_umap_{color}.png"


def save_method_umap_pngs(
    adata,
    *,
    data_path: str | Path,
    method_slug: str,
    basis_key: str = "umap",
    figsize: tuple[float, float] = DEFAULT_UMAP_FIGSIZE,
    png_dpi: int = PNG_DPI,
    point_size: float | None = 1.0,
    palette=None,
    cmap=None,
    alpha: float = 1.0,
    transparent: bool = True,
    colors: tuple[str, ...] = DEFAULT_UMAP_COLORS,
) -> None:
    for color in colors:
        plot_points_only_png(
            adata,
            basis_key=basis_key,
            color=color,
            output_path=_umap_output_path(
                data_path=data_path,
                method_slug=method_slug,
                color=color,
            ),
            figsize=figsize,
            png_dpi=png_dpi,
            point_size=point_size,
            palette=palette,
            cmap=cmap,
            alpha=alpha,
            transparent=transparent,
        )
# %%
save_dir = os.path.join(CFG["benchmark"]["dlpfc"]["results_dir"], "umap_plots")
sc.pp.neighbors(adata, use_rep='Unintegrated', n_neighbors=15)
sc.tl.umap(adata, min_dist=0.3, spread=1.0)
# %%
with mpl.rc_context({
        "figure.dpi": PNG_DPI,
        "savefig.dpi": PNG_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "Arial",
    }):
    fig, ax = plt.subplots(figsize=DEFAULT_UMAP_FIGSIZE)
    sc.pl.umap(adata, color="section", ax=ax, show=False)
    fig.savefig(
        os.path.join(save_dir, "Unintegrated_umap_section.pdf"),
        dpi=PNG_DPI,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    fig, ax = plt.subplots(figsize=DEFAULT_UMAP_FIGSIZE)
    sc.pl.umap(adata, color="layer_guess", ax=ax, show=False)
    fig.savefig(
        os.path.join(save_dir, "Unintegrated_umap_layer_guess.pdf"),
        dpi=PNG_DPI,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

# %%
benchmark_keys = ["X_STAligner", 
                  "X_paste", "X_GraphST", "X_prime", 
                  "X_precast", "Unintegrated"]
for method in benchmark_keys:
    method_slug = method.replace("X_", "")
    sc.pp.neighbors(adata, use_rep=method, n_neighbors=15)
    sc.tl.umap(adata, min_dist=0.3, spread=1.0)
    save_method_umap_pngs(adata,
        colors=("section", "layer_guess"),
        data_path=save_dir, method_slug=method_slug)
# %%
