"""Human immune data: integration with every method, scIB benchmark, results table and UMAPs.

Cell-structured script (`#%%` cells), written to be run top to bottom or cell by cell. Reads
`Immune_ALL_human.h5ad` (Luecken et al. 2022; log-normalised X, counts layer) from config.yaml
`benchmark.immune.dir`.

  1. 2,000 batch-aware HVGs (cell_ranger flavour), PCA (30) = the unintegrated reference
  2. Scanorama, Harmony, MNN, ComBat, BBKNN, scVI and PRIME (`prime.ensemble_mnn_correct`), each followed by a UMAP
     (15 neighbours, min_dist 0.3) saved as point-only PNGs
  3. the embeddings of the Seurat-based methods (CCA, FastMNN, joint PCA, RPCA) written by
     `../r_methods/run_r_methods.R` are attached
  4. scIB benchmark (scib-metrics Benchmarker: Leiden NMI / ARI, cLISI, BRAS, iLISI, kBET, PCR comparison) plus the
     isolated-label silhouette; aggregates Bio = mean of the bio metrics, Total = 0.6 Bio + 0.4 Batch
  5. results table (PDF) and the UMAP coordinates of selected methods

Writes into the same folder: Immune_ALL_human_all_embeddings.h5ad (all embeddings in obsm), benchmark_results.csv,
benchmark_results_with_isolated_labels.csv, benchmark_results_table.pdf, umap_plots/, umap_embeddings/.
"""
#%% import libraries
import logging
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples

#%% configuration
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # pipeline/ (common.py)
import common as C  # noqa: E402

CFG = C.load_config()

#%% Save the umap plot

PNG_DPI = 900
PDF_FONT_FAMILY = "Arial"
PDF_FONT_FALLBACK = "DejaVu Sans"
VECTOR_WARN_N_OBS = 50000
EXPORT_FULL_SVG = False
PDF_FONTTYPE = 42
PS_FONTTYPE = 42
SVG_FONTTYPE = "none"
DEFAULT_UMAP_FIGSIZE = (4, 4)
DEFAULT_UMAP_COLORS = ("batch", "cell_type")
RESULTS_METRIC_TYPE_ROW = "Metric Type"
AGGREGATE_SCORE_VALUE = "Aggregate score"
BIO_CONSERVATION_COL = "Bio conservation"
BATCH_CORRECTION_COL = "Batch correction"
TOTAL_SCORE_COL = "Total"
ISOLATED_LABELS_COL = "Isolated labels"

logger = logging.getLogger(__name__)

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

#%%
data_path = CFG["benchmark"]["immune"]["dir"]

adata = sc.read(
    f"{data_path}/Immune_ALL_human.h5ad",
)
adata.obs["cell_type"] = adata.obs["final_annotation"]
adata

#%%
plt.rcParams['font.size'] = 16
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key="batch")
sc.tl.pca(adata, n_comps=30, use_highly_variable=True)
# adata = adata[:, adata.var.highly_variable].copy()
adata.obsm["Unintegrated"] = adata.obsm["X_pca"]

#%%
sc.pp.neighbors(adata)
sc.tl.umap(adata, min_dist=0.3, spread=1.0)
# sc.pl.umap(adata, 
#            color=["batch", "cell_type"], 
#            wspace=0.3, 
#            frameon=False, 
#            ncols=1
#         )
save_method_umap_pngs(adata, data_path=data_path, method_slug="unintegrated")
#%%
with plt.rc_context(
        {"figure.dpi": 300,
         "pdf.fonttype": 42,
         "ps.fonttype": 42,
         "font.family": "Arial"},
    ):
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    sc.pl.umap(adata, color=["batch"], 
               wspace=0.3, frameon=False, ncols=1, 
               show=False, ax=ax[0])
    sc.pl.umap(adata, color=["cell_type"], 
               wspace=0.3, frameon=False, ncols=1, 
               show=False, ax=ax[1])
    plt.tight_layout()
    plt.savefig(f"{data_path}/umap_plots/unintegrated_umap_combined.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

#%% scanorama
import scanorama

# List of adata per batch
batch_cats = adata.obs.batch.cat.categories
adata_list = [adata[adata.obs.batch == b].copy() for b in batch_cats]
scanorama.integrate_scanpy(adata_list, dimred=100, hvg=2000)

adata.obsm["Scanorama"] = np.zeros((adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1]))
for i, b in enumerate(batch_cats):
    adata.obsm["Scanorama"][adata.obs.batch == b] = adata_list[i].obsm["X_scanorama"]

sc.pp.neighbors(adata, use_rep="Scanorama")
sc.tl.umap(adata, min_dist=0.3, spread=1.0)

save_method_umap_pngs(adata, data_path=data_path, method_slug="scanorama")

#%% harmony
from harmonypy import run_harmony

harmony_out = run_harmony(adata.obsm["X_pca"], adata.obs, "batch")
adata.obsm["Harmony"] = harmony_out.Z_corr

sc.pp.neighbors(adata, use_rep="Harmony")
sc.tl.umap(adata, min_dist=0.3, spread=1.0)

save_method_umap_pngs(adata, data_path=data_path, method_slug="harmony")

#%% MNN
batch_cats = adata.obs.batch.cat.categories
adata_list = [adata[adata.obs.batch == b].copy() for b in batch_cats]
hvgs = adata.var[adata.var.highly_variable].index.tolist()

corrected = sc.external.pp.mnn_correct(
    adata_list,
    var_subset=hvgs,
    batch_key='batch',
    # batch_categories=['batch1', 'batch2', 'batch3'],
    k=20,
    sigma=0.1,
    svd_dim=100,  # Number of SVD dimensions
    save_raw=True,  # Save uncorrected data in .raw
    do_concatenate=True,  # Return concatenated object
    n_jobs=16,
)

adata_corrected = ad.concat(corrected[0][0])
adata_corrected = adata_corrected[adata.obs_names, :]
adata.obsm["MNN"] = adata_corrected.X.toarray()

sc.pp.neighbors(adata, use_rep="MNN")
sc.tl.umap(adata, min_dist=0.3, spread=1.0)

save_method_umap_pngs(adata, data_path=data_path, method_slug="mnn")

#%% combat
adata_combat_expression = adata.copy()
adata_combat_expression = sc.pp.combat(adata, key="batch", inplace=False)

adata_combat = ad.AnnData(X=adata_combat_expression)
adata_combat.obs = adata.obs.copy()
adata_combat.var = adata.var.copy()

adata.obsm["ComBat"] = adata_combat.X

sc.pp.neighbors(adata, use_rep="ComBat")
sc.tl.umap(adata, min_dist=0.3, spread=1.0)

save_method_umap_pngs(adata, data_path=data_path, method_slug="combat")

#%% bbknn
import bbknn

adata_highly_var = adata[:, adata.var.highly_variable].copy()
neighbors_within_batch = 3
adata_bbknn = adata_highly_var.copy()
adata_highly_var.layers["logcounts"] = adata_highly_var.X.copy()
adata_bbknn.X = adata_highly_var.layers["logcounts"].copy()
sc.pp.pca(adata_bbknn)
bbknn.bbknn(
    adata_bbknn, batch_key="batch", neighbors_within_batch=neighbors_within_batch
)
sc.tl.umap(adata_bbknn)

save_method_umap_pngs(adata_bbknn, data_path=data_path, method_slug="bbknn")

adata.obsm["BBKNN"] = adata_bbknn.X.toarray().copy()

#%% scVI
import scvi

adata_scvi = adata_highly_var.copy()
scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts", batch_key="batch")
model_scvi = scvi.model.SCVI(adata_scvi)
max_epochs_scvi = np.min([round((20000 / adata_scvi.n_obs) * 400), 400])
model_scvi.train()
adata_scvi.obsm["X_scVI"] = model_scvi.get_latent_representation()
sc.pp.neighbors(adata_scvi, use_rep="X_scVI")
sc.tl.umap(adata_scvi)

adata.obsm["scVI"] = adata_scvi.obsm["X_scVI"].copy()

save_method_umap_pngs(adata_scvi, data_path=data_path, method_slug="scvi")

#%% PRIME
from prime import ensemble_mnn_correct

adata_highly_var.obsm["PRIME"] = ensemble_mnn_correct(
    adata_highly_var,
    batch_key="batch",
    n_projections=4,
    target_dim=128,
    k_neighbors=15,
    consensus_threshold=0.4,
    sigma=0.1,
    random_state=42,
    key_added=None,
    inplace=False
)

sc.pp.neighbors(adata_highly_var, use_rep="PRIME")
sc.tl.umap(adata_highly_var, min_dist=0.3, spread=1.0)
adata.obsm["PRIME"] = adata_highly_var.obsm["PRIME"].copy()

save_method_umap_pngs(adata_highly_var, data_path=data_path, method_slug="prime")

#%% load cca
import numpy as np

cca_embedding = np.load(
    f"{data_path}/Immune_ALL_human_cca_corrected.npy"
    )

adata.obsm["CCA"] = cca_embedding
sc.pp.neighbors(adata, use_rep="CCA")
sc.tl.umap(adata, min_dist=0.3, spread=1.0)

save_method_umap_pngs(adata, data_path=data_path, method_slug="cca")

#%% load fastmnn
fastmnn_embedding = np.load(
    f"{data_path}/Immune_ALL_human_fastmnn_corrected.npy"
    )

adata.obsm["FastMNN"] = fastmnn_embedding
sc.pp.neighbors(adata, use_rep="FastMNN")
sc.tl.umap(adata, min_dist=0.3, spread=1.0)

save_method_umap_pngs(adata, data_path=data_path, method_slug="fastmnn")

#%% load jpca
jpca_embedding = np.load(
    f"{data_path}/Immune_ALL_human_jpca_corrected.npy"
    )

adata.obsm["jPCA"] = jpca_embedding
sc.pp.neighbors(adata, use_rep="jPCA")
sc.tl.umap(adata, min_dist=0.3, spread=1.0)

save_method_umap_pngs(adata, data_path=data_path, method_slug="jpca")

#%% load rpca
rpca_embedding = np.load(
    f"{data_path}/Immune_ALL_human_rpca_corrected.npy"
    )

adata.obsm["RPCA"] = rpca_embedding
sc.pp.neighbors(adata, use_rep="RPCA")
sc.tl.umap(adata, min_dist=0.3, spread=1.0)

save_method_umap_pngs(adata, data_path=data_path, method_slug="rpca")

#%% save the adata with all embeddings
save_file = f"{data_path}/Immune_ALL_human_all_embeddings.h5ad"
adata.write(save_file)

#%%
def _get_isolated_labels(
    labels: np.ndarray,
    batch: np.ndarray | None,
    iso_threshold: int | None,
) -> np.ndarray:
    if batch is None:
        return np.unique(labels)

    tmp = pd.DataFrame({"label": labels, "batch": batch}).drop_duplicates()
    batch_per_label = tmp.groupby("label")["batch"].count()

    if iso_threshold is None:
        iso_threshold = int(batch_per_label.min())

    logger.info("Isolated label threshold: <= %s batch(es) per label", iso_threshold)
    isolated = batch_per_label[batch_per_label <= iso_threshold].index.to_numpy()

    if isolated.size == 0:
        logger.warning("No isolated labels found with threshold=%s", iso_threshold)

    return isolated


def isolated_label_score_single(
    X: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray | None = None,
    rescale: bool = True,
    iso_threshold: int | None = None,
) -> float:
    isolated = _get_isolated_labels(labels, batch, iso_threshold)
    if isolated.size == 0:
        logger.warning("No isolated labels found. Returning NaN.")
        return float("nan")

    try:
        sil_all = silhouette_samples(X, labels, metric="euclidean")
    except ValueError as exc:
        logger.warning("Failed to compute isolated label score: %s", exc)
        return float("nan")

    if rescale:
        sil_all = (sil_all + 1.0) / 2.0

    per_label_scores = []
    for label in isolated:
        mask = labels == label
        per_label_scores.append(float(np.mean(sil_all[mask])))

    return float(np.mean(per_label_scores))


def compute_isolated_label_scores(
    adata: ad.AnnData,
    label_key: str,
    embedding_keys: list[str],
    batch_key: str | None = None,
    rescale: bool = True,
    iso_threshold: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    if label_key not in adata.obs.columns:
        raise ValueError(f"label_key={label_key!r} not found in adata.obs")
    if batch_key is not None and batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key={batch_key!r} not found in adata.obs")

    labels = adata.obs[label_key].to_numpy().astype(str)
    batch = adata.obs[batch_key].to_numpy().astype(str) if batch_key else None

    results = {}
    for key in embedding_keys:
        if key not in adata.obsm:
            logger.warning("Embedding key %r not found in adata.obsm", key)
            results[key] = float("nan")
            continue

        X = adata.obsm[key]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

        if verbose:
            print(f"[{key}] Computing isolated label score...")

        score = isolated_label_score_single(
            X=X,
            labels=labels,
            batch=batch,
            rescale=rescale,
            iso_threshold=iso_threshold,
        )
        results[key] = score

        if verbose:
            print(f"  score = {score:.4f}")

    return pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=[ISOLATED_LABELS_COL],
    )


def _merge_isolated_label_scores(
    results_df: pd.DataFrame,
    isolated_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = results_df.copy()
    merged[ISOLATED_LABELS_COL] = np.nan

    method_rows = merged.index[merged.index != RESULTS_METRIC_TYPE_ROW]
    merged.loc[method_rows, ISOLATED_LABELS_COL] = (
        isolated_df.reindex(method_rows)[ISOLATED_LABELS_COL].to_numpy()
    )

    if RESULTS_METRIC_TYPE_ROW in merged.index:
        merged.loc[RESULTS_METRIC_TYPE_ROW, ISOLATED_LABELS_COL] = BIO_CONSERVATION_COL
        cols_without_new = [col for col in merged.columns if col != ISOLATED_LABELS_COL]
        metric_type_row = merged.loc[RESULTS_METRIC_TYPE_ROW, cols_without_new]
        aggregate_cols = metric_type_row.index[
            metric_type_row == AGGREGATE_SCORE_VALUE
        ].tolist()
        insert_at = (
            cols_without_new.index(aggregate_cols[0])
            if aggregate_cols
            else len(cols_without_new)
        )
        reordered_cols = (
            cols_without_new[:insert_at]
            + [ISOLATED_LABELS_COL]
            + cols_without_new[insert_at:]
        )
        merged = merged.loc[:, reordered_cols]

    return merged


def _recompute_aggregate_scores(results_df: pd.DataFrame) -> pd.DataFrame:
    if RESULTS_METRIC_TYPE_ROW not in results_df.index:
        raise ValueError("results_df must include the Metric Type row.")

    updated = results_df.copy()
    metric_type_row = updated.loc[RESULTS_METRIC_TYPE_ROW]
    method_rows = updated.index[updated.index != RESULTS_METRIC_TYPE_ROW]

    bio_metric_cols = metric_type_row.index[
        metric_type_row == BIO_CONSERVATION_COL
    ].tolist()
    bio_metrics = updated.loc[method_rows, bio_metric_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )
    batch_scores = pd.to_numeric(
        updated.loc[method_rows, BATCH_CORRECTION_COL],
        errors="coerce",
    )

    updated.loc[method_rows, BIO_CONSERVATION_COL] = bio_metrics.mean(axis=1)
    updated.loc[method_rows, TOTAL_SCORE_COL] = (
        0.6 * pd.to_numeric(updated.loc[method_rows, BIO_CONSERVATION_COL], errors="coerce")
        + 0.4 * batch_scores
    )

    updated.loc[RESULTS_METRIC_TYPE_ROW, BATCH_CORRECTION_COL] = AGGREGATE_SCORE_VALUE
    updated.loc[RESULTS_METRIC_TYPE_ROW, BIO_CONSERVATION_COL] = AGGREGATE_SCORE_VALUE
    updated.loc[RESULTS_METRIC_TYPE_ROW, TOTAL_SCORE_COL] = AGGREGATE_SCORE_VALUE
    return updated


#%% benchmarks
# load the adata with all embeddings
data_path = CFG["benchmark"]["immune"]["dir"]

adata = sc.read(f"{data_path}/Immune_ALL_human_all_embeddings.h5ad")

from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

benchmark_keys = ["ComBat", "Scanorama", "Harmony", "MNN", 
                  "PRIME", "BBKNN", "scVI", "Unintegrated",
                  "CCA", "FastMNN", "jPCA", "RPCA"]
# benchmark_keys = ["Unintegrated", "Scanorama"]

bm = Benchmarker(
    adata,
    batch_key="batch",
    label_key="cell_type",
    bio_conservation_metrics=BioConservation(
            nmi_ari_cluster_labels_leiden=True, 
            nmi_ari_cluster_labels_kmeans=False, 
            silhouette_label=False,
            isolated_labels=False,
        ),
    batch_correction_metrics=BatchCorrection(graph_connectivity=False),
    embedding_obsm_keys=benchmark_keys,
    n_jobs=1,
)
bm.benchmark()

#%% plot the results and save the results table
bm.plot_results_table()

res_df_file = f"{data_path}/benchmark_results.csv"
bm.get_results().to_csv(res_df_file, index=True)

#%% load the results table
results_df = pd.read_csv(res_df_file, index_col=0)
results_df
#%%
isolated_results_df = compute_isolated_label_scores(
    adata,
    label_key="cell_type",
    embedding_keys=benchmark_keys,
    batch_key="batch",
    rescale=True,
    iso_threshold=None,
    verbose=True,
)
results_df = _merge_isolated_label_scores(results_df, isolated_results_df)
results_df = _recompute_aggregate_scores(results_df)

revised_res_df_file = f"{data_path}/benchmark_results_with_isolated_labels.csv"
results_df.to_csv(revised_res_df_file, index=True)

#%% plot the results table
from contextlib import nullcontext
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Colormap
from matplotlib.text import Text
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar

METRIC_TYPE_ROW = "Metric Type"
AGGREGATE_SCORE = "Aggregate score"
DEFAULT_SORT_PRIORITY = ("Total", "Batch correction", "Bio conservation")
OUTPUT_DIR = Path(CFG["benchmark"]["immune"]["dir"])
PDF_FILENAME = "benchmark_results_pub.pdf"
FIGSIZE = (7.0, 4.2)
PDF_FONT_FAMILY = "Arial"
PDF_FONTTYPE = 42
PS_FONTTYPE = 42
BBOX_INCHES = "tight"
PAD_INCHES = 0.02
CELL_CMAP = mpl.cm.PRGn
SCORE_CMAP = mpl.cm.YlGnBu
SORT_COL = "Total"

__all__ = [
    "AGGREGATE_SCORE",
    "BBOX_INCHES",
    "CELL_CMAP",
    "FIGSIZE",
    "METRIC_TYPE_ROW",
    "OUTPUT_DIR",
    "PAD_INCHES",
    "PDF_FILENAME",
    "PDF_FONT_FAMILY",
    "PDF_FONTTYPE",
    "PS_FONTTYPE",
    "SCORE_CMAP",
    "SORT_COL",
    "plot_scib_results_table_repro",
    "save_scib_results_publication_pdf",
]


def _validate_results_frame(df: pd.DataFrame) -> None:
    if METRIC_TYPE_ROW not in df.index:
        raise ValueError(
            "Input DataFrame must include a 'Metric Type' row, as returned by "
            "Benchmarker.get_results()."
        )


def _resolve_score_columns(df: pd.DataFrame) -> list[str]:
    metric_type_row = df.loc[METRIC_TYPE_ROW]
    score_cols = metric_type_row.index[metric_type_row == AGGREGATE_SCORE].tolist()
    if not score_cols:
        raise ValueError(
            "Input DataFrame must include at least one aggregate score column where "
            "df.loc['Metric Type', col] == 'Aggregate score'."
        )
    return score_cols


def _resolve_sort_col(score_cols: list[str], sort_col: str | None) -> str:
    if sort_col is not None:
        if sort_col not in score_cols:
            raise ValueError(
                f"sort_col {sort_col!r} must be one of the aggregate score columns: "
                f"{score_cols!r}."
            )
        return sort_col

    for candidate in DEFAULT_SORT_PRIORITY:
        if candidate in score_cols:
            return candidate
    return score_cols[0]


def _build_cell_cmap(
    plot_df: pd.DataFrame,
    col: str,
    cell_cmap: Colormap,
    cell_num_stds: float,
):
    return normed_cmap(plot_df[col], cmap=cell_cmap, num_stds=cell_num_stds)


def _default_figsize(df: pd.DataFrame, num_methods: int) -> tuple[float, float]:
    return (len(df.columns) * 1.25, 3 + 0.3 * num_methods)


def _set_figure_text_font_family(fig: Figure, font_family: str) -> None:
    for text_artist in fig.findobj(match=Text):
        text_artist.set_fontfamily(font_family)


def plot_scib_results_table_repro(
    df: pd.DataFrame,
    cell_cmap: Colormap = mpl.cm.PRGn,
    score_cmap: Colormap = mpl.cm.YlGnBu,
    cell_num_stds: float = 2.5,
    sort_col: str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 300,
    figsize: tuple[float, float] | None = None,
    svg_text_as_text: bool = True,
) -> tuple[Figure, Axes, Table]:
    """Reproduce scib-metrics' benchmark results table without using its plotting API.

    Parameters
    ----------
    df
        Results table with the same structure as ``Benchmarker.get_results()``.
        Rows are methods plus one ``"Metric Type"`` row.
    cell_cmap
        Colormap used for the metric cells.
    score_cmap
        Colormap used for aggregate score bars.
    cell_num_stds
        Passed through to ``plottable.cmap.normed_cmap`` for metric-cell scaling.
    sort_col
        Aggregate column used for sorting. Defaults to ``Total``, then
        ``Batch correction``, then ``Bio conservation``; otherwise the first
        aggregate column.
    show
        Whether to call ``plt.show()``.
    save_path
        Optional target path for saving the figure. Any format supported by
        Matplotlib is accepted.
    dpi
        Save DPI when ``save_path`` is provided.
    figsize
        Optional explicit figure size. Defaults to the same shape heuristic used
        by ``scib-metrics``.
    svg_text_as_text
        When ``True``, save SVG text as editable text objects by using
        ``svg.fonttype = 'none'`` during figure creation.
    """

    _validate_results_frame(df)

    results_df = df.copy()
    score_cols = _resolve_score_columns(results_df)
    resolved_sort_col = _resolve_sort_col(score_cols, sort_col)

    plot_df = results_df.drop(index=METRIC_TYPE_ROW)
    plot_df = plot_df.sort_values(by=resolved_sort_col, ascending=False).astype(np.float64)
    plot_df["Method"] = plot_df.index

    metric_type_row = results_df.loc[METRIC_TYPE_ROW]
    other_cols = metric_type_row.index[metric_type_row != AGGREGATE_SCORE].tolist()
    num_methods = plot_df.shape[0]

    column_definitions = [
        ColumnDefinition(
            "Method",
            width=1.5,
            textprops={"ha": "left", "weight": "bold"},
        )
    ]
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=1,
            textprops={
                "ha": "center",
                "bbox": {"boxstyle": "circle", "pad": 0.25},
            },
            cmap=_build_cell_cmap(plot_df, col, cell_cmap=cell_cmap, cell_num_stds=cell_num_stds),
            group=metric_type_row[col],
            formatter="{:.2f}",
        )
        for col in other_cols
    ]
    column_definitions += [
        ColumnDefinition(
            col,
            width=1,
            title=col.replace(" ", "\n", 1),
            plot_fn=bar,
            plot_kw={
                "cmap": score_cmap,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
            },
            group=metric_type_row[col],
            border="left" if i == 0 else None,
        )
        for i, col in enumerate(score_cols)
    ]

    rc_context = mpl.rc_context({"svg.fonttype": "none"}) if svg_text_as_text else nullcontext()
    with rc_context:
        fig, ax = plt.subplots(figsize=figsize or _default_figsize(results_df, num_methods))
        table = Table(
            plot_df,
            cell_kw={"linewidth": 0, "edgecolor": "k"},
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, facecolor=ax.get_facecolor(), dpi=dpi)

    if show:
        plt.show()

    return fig, ax, table


def save_scib_results_publication_pdf(
    fig: Figure,
    output_path: str | Path,
    *,
    font_family: str = PDF_FONT_FAMILY,
    bbox_inches: str | None = BBOX_INCHES,
    pad_inches: float = PAD_INCHES,
) -> None:
    """Save an existing figure as a publication-ready PDF with editable text."""

    pdf_path = Path(output_path)
    if pdf_path.suffix.lower() != ".pdf":
        pdf_path = pdf_path.with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    _set_figure_text_font_family(fig, font_family)
    with mpl.rc_context(
        {
            "pdf.fonttype": PDF_FONTTYPE,
            "ps.fonttype": PS_FONTTYPE,
            "font.family": font_family,
        }
    ):
        fig.savefig(
            pdf_path,
            format="pdf",
            facecolor=fig.get_facecolor(),
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
#%%
# re-order the Isolated labels column to the fourth column
results_df_tmp = results_df.copy()
col = results_df_tmp.pop(ISOLATED_LABELS_COL)
results_df_tmp.insert(3, ISOLATED_LABELS_COL, col)
results_df_tmp
results_df = results_df_tmp



#%% plot the results table with the custom function and save the figure
import matplotlib as mpl

table_fig_file = f"{data_path}/benchmark_results_table.pdf"
with mpl.rc_context({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "Arial",   # Editable text in the exported PDF.
    "figure.dpi": 300,
}):
    fig, ax, table = plot_scib_results_table_repro(
        results_df,
        cell_cmap=mpl.cm.PRGn,
        score_cmap=mpl.cm.OrRd,
        sort_col="Total",
        figsize=(14, 8),
        show=False,
        save_path=None,
    )

    fig.savefig(
        table_fig_file,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )

# %%
fig
# %% UMAP of the PRIME embedding
data_dir = CFG["benchmark"]["immune"]["dir"]
data_file = "Immune_ALL_human_all_embeddings.h5ad"

adata = sc.read(f"{data_dir}/{data_file}")

sc.pp.neighbors(adata, use_rep="PRIME")
sc.tl.umap(adata, min_dist=0.3, spread=1.0)

umap_embedding = adata.obsm["X_umap"]
os.makedirs(f"{data_dir}/umap_embeddings", exist_ok=True)
np.save(f"{data_dir}/umap_embeddings/Immune_ALL_human_prime_umap.npy", umap_embedding)


# %% loading the saved adata
data_dir = CFG["benchmark"]["immune"]["dir"]
adata = sc.read(f"{data_dir}/Immune_ALL_human_all_embeddings.h5ad")
adata.obsm.keys()

# %%
methods_selected = ["Unintegrated", "Scanorama", "Harmony", "MNN",]
save_dir = f"{data_dir}/umap_embeddings"
os.makedirs(save_dir, exist_ok=True)

for method in methods_selected:
    sc.pp.neighbors(adata, use_rep=method)
    sc.tl.umap(adata, min_dist=0.3, spread=1.0)
    
    adata.obsm[f"{method}_umap"] = adata.obsm["X_umap"].copy()
    np.save(f"{save_dir}/Immune_ALL_human_{method.lower()}_umap.npy", adata.obsm[f"{method}_umap"])
# %%
