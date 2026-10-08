"""Pseudotime validation of the haematopoietic lineages (human immune trajectory panels): lineage filter and ordering
metrics, per-gene pseudotime trends and module radar, each as a function of the lineage (erythroid, monocyte).
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.stats import kendalltau, spearmanr

try:
    from statsmodels.stats.multitest import multipletests

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

PSEUDOTIME_COL = "pseudotime"
CELLTYPE_COL = "cell_type"
BRANCH_COL = None
LINEAGE_CONFIGS = {
    "erythroid": {
        "target_celltypes": ["HSPCs", "Erythroid progenitors", "Erythrocytes"],
        "stage_rank": {"HSPCs": 1, "Erythroid progenitors": 2, "Erythrocytes": 3},
        "reference_modules": {
            "stemness_down": ["HLF", "MECOM", "MLLT3", "MEIS1", "SPINK2", "CD34", "KIT", "FLT3"],
            "erythroid_entry": ["KLF1", "EPOR", "TFRC", "GATA1", "TAL1", "AHSP", "ALAS2"],
            "erythroid_mid": ["TFRC", "AHSP", "ALAS2", "FECH", "GYPA", "CA1", "ANK1", "SPTA1", "SPTB", "TMOD1"],
            "erythroid_terminal": ["GYPA", "SLC4A1", "RHAG", "HBA1", "HBA2", "HBB", "BPGM", "FECH", "ANK1", "TMOD1"],
        },
        "expected_gene_direction": {
            "HLF": -1, "MECOM": -1, "MLLT3": -1, "MEIS1": -1, "SPINK2": -1, "CD34": -1, "KIT": -1, "FLT3": -1,
            "KLF1": +1, "EPOR": +1, "TFRC": +1, "GATA1": +1, "TAL1": +1, "AHSP": +1, "ALAS2": +1,
            "FECH": +1, "GYPA": +1, "CA1": +1, "ANK1": +1, "SPTA1": +1, "SPTB": +1, "TMOD1": +1,
            "SLC4A1": +1, "RHAG": +1, "HBA1": +1, "HBA2": +1, "HBB": +1, "BPGM": +1,
        },
        "target_modules": ["stemness_down", "erythroid_entry", "erythroid_mid", "erythroid_terminal"],
        "representative_genes_for_plot": ["HLF", "MECOM", "KLF1", "EPOR", "TFRC", "ALAS2", "GYPA", "HBA1", "HBB"],
    },
    "monocyte": {
        "target_celltypes": ["HSPCs", "Monocyte progenitors", "CD14+ Monocytes", "CD16+ Monocytes", ], # "Monocyte-derived dendritic cells"
        "stage_rank": {"HSPCs": 1, "Monocyte progenitors": 2, "CD14+ Monocytes": 3, "CD16+ Monocytes": 4, }, # "Monocyte-derived dendritic cells": 4
        "reference_modules": {
            "stemness_down": ["HLF", "MECOM", "MLLT3", "MEIS1", "SPINK2", "CD34", "KIT", "FLT3"],
            "myeloid_entry": ["LYZ", "MPO", "ELANE", "AZU1", "PRTN3", "CSF3R", "SPI1", "CTSG"],
            "monocyte_prog": ["FCN1", "S100A8", "S100A9", "CTSS", "TYMP", "CSF1R"],
            "monocyte_late": ["LST1", "FCER1G", "IFI30", "CTSS", "TYMP", "SAT1", "CTSD"],
            "mo_dc_extension": ["FCER1A", "CLEC10A", "CD1C", "HLA-DRA", "CD74"],
        },
        "expected_gene_direction": {
            "HLF": -1, "MECOM": -1, "MLLT3": -1, "MEIS1": -1, "SPINK2": -1, "CD34": -1, "KIT": -1, "FLT3": -1,
            "LYZ": +1, "MPO": +1, "ELANE": +1, "AZU1": +1, "PRTN3": +1, "CSF3R": +1, "SPI1": +1, "CTSG": +1,
            "FCN1": +1, "S100A8": +1, "S100A9": +1, "CTSS": +1, "TYMP": +1, "CSF1R": +1,
            "LST1": +1, "FCER1G": +1, "IFI30": +1, "SAT1": +1, "CTSD": +1,
            "FCER1A": +1, "CLEC10A": +1, "CD1C": +1, "HLA-DRA": +1, "CD74": +1,
        },
        "target_modules": ["stemness_down", "myeloid_entry", "monocyte_prog", "monocyte_late", "mo_dc_extension"],
        "representative_genes_for_plot": ["HLF", "MECOM", "LYZ", "CSF3R", "FCN1", "S100A8", "S100A9", "LST1", "FCER1G", "FCER1A"],
    },
}
N_BINS = 30


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def check_normalization(adata: ad.AnnData) -> None:
    """
    Roughly check whether adata.X appears normalized rather than raw counts.
    Raw counts are usually integer-valued and can have a much larger maximum.
    Print a warning if the matrix looks like unnormalized input.
    """
    sample = adata.X[:200, :]
    if issparse(sample):
        sample = sample.toarray()
    sample = np.asarray(sample, dtype=float)
    max_val = sample.max()
    is_integer = np.allclose(sample, sample.astype(int), atol=1e-3)
    if is_integer and max_val > 50:
        print(
            f"[WARNING] adata.X may contain raw counts (max={max_val:.1f}, integer-valued)."
            " Run sc.pp.normalize_total + sc.pp.log1p before this script."
            " Correlation-based analysis is not reliable on raw counts."
        )
    else:
        print(f"[INFO] adata.X check passed (max={max_val:.4f}, likely normalized).")


def get_dense_vector(adata: ad.AnnData, gene: str) -> np.ndarray:
    idx = adata.var_names.get_loc(gene)
    values = adata.X[:, idx]
    if issparse(values):
        values = values.toarray().ravel()
    else:
        values = np.asarray(values).ravel()
    return values


def build_expected_gene_direction(
    modules: Dict[str, List[str]],
    module_directions: Dict[str, int],
) -> Dict[str, int]:
    gene_directions: Dict[str, int] = {}
    for module_name, genes in modules.items():
        if module_name not in module_directions:
            raise ValueError(f"Missing direction for module '{module_name}'.")
        direction = module_directions[module_name]
        for gene in genes:
            if gene in gene_directions and gene_directions[gene] != direction:
                raise ValueError(f"Gene '{gene}' has conflicting expected directions.")
            gene_directions[gene] = direction
    return gene_directions


def compute_stage_order_metrics(
    adata: ad.AnnData,
    celltype_col: str,
    pseudotime_col: str,
    stage_rank: Dict[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["cell_type", "stage_rank", "n_cells", "median_pseudotime", "mean_pseudotime"]
    rows = []
    available_celltypes = set(adata.obs[celltype_col].astype(str))

    for cell_type, rank in stage_rank.items():
        if cell_type not in available_celltypes:
            continue
        values = adata.obs.loc[adata.obs[celltype_col] == cell_type, pseudotime_col].dropna().to_numpy()
        # remove inf and na value
        values = values[np.isfinite(values)]

        if values.size == 0:
            continue
        rows.append(
            {
                "cell_type": cell_type,
                "stage_rank": rank,
                "n_cells": int(values.size),
                "median_pseudotime": float(np.median(values)),
                "mean_pseudotime": float(np.mean(values)),
            }
        )

    stage_df = pd.DataFrame(rows, columns=columns)
    if not stage_df.empty:
        stage_df = stage_df.sort_values("stage_rank").reset_index(drop=True)

    if len(stage_df) >= 2:
        rho, rho_p = spearmanr(stage_df["stage_rank"], stage_df["median_pseudotime"])
        tau, tau_p = kendalltau(stage_df["stage_rank"], stage_df["median_pseudotime"])
    else:
        rho, rho_p, tau, tau_p = np.nan, np.nan, np.nan, np.nan

    summary_df = pd.DataFrame(
        [
            {
                "metric": "stage_order_consistency",
                "spearman_rho": rho,
                "spearman_p": rho_p,
                "kendall_tau": tau,
                "kendall_p": tau_p,
            }
        ]
    )
    return stage_df, summary_df


def compute_gene_trend_metrics(
    adata: ad.AnnData,
    pseudotime_col: str,
    expected_direction: Dict[str, int],
) -> pd.DataFrame:
    pt = adata.obs[pseudotime_col].to_numpy(dtype=float)
    rows = []

    for gene, direction in expected_direction.items():
        row = {
            "gene": gene,
            "present": gene in adata.var_names,
            "expected_direction": direction,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "kendall_tau": np.nan,
            "kendall_p": np.nan,
            "signed_spearman_rho": np.nan,
            "signed_kendall_tau": np.nan,
        }
        if row["present"]:
            expr = get_dense_vector(adata, gene)
            rho, rho_p = spearmanr(expr, pt)
            tau, tau_p = kendalltau(expr, pt)
            row.update(
                {
                    "spearman_rho": rho,
                    "spearman_p": rho_p,
                    "kendall_tau": tau,
                    "kendall_p": tau_p,
                    "signed_spearman_rho": direction * rho if not np.isnan(rho) else np.nan,
                    "signed_kendall_tau": direction * tau if not np.isnan(tau) else np.nan,
                }
            )
        rows.append(row)

    metric_df = pd.DataFrame(rows)
    metric_df["spearman_p_adj"] = np.nan

    if HAS_STATSMODELS:
        present_mask = metric_df["present"] & metric_df["spearman_p"].notna()
        if present_mask.any():
            _, p_adj, _, _ = multipletests(metric_df.loc[present_mask, "spearman_p"], method="fdr_bh")
            metric_df.loc[present_mask, "spearman_p_adj"] = p_adj
    else:
        print("[INFO] statsmodels is not installed; skipping FDR correction.")

    return metric_df


def summarize_module_trend_metrics(
    gene_metric_df: pd.DataFrame,
    modules: Dict[str, List[str]],
) -> pd.DataFrame:
    rows = []
    for module_name, genes in modules.items():
        subset = gene_metric_df[gene_metric_df["gene"].isin(genes)]
        rows.append(
            {
                "module": module_name,
                "n_genes_present": int(subset["present"].sum()) if not subset.empty else 0,
                "median_signed_spearman_rho": subset["signed_spearman_rho"].median(),
                "median_signed_kendall_tau": subset["signed_kendall_tau"].median(),
                "mean_signed_spearman_rho": subset["signed_spearman_rho"].mean(),
                "mean_signed_kendall_tau": subset["signed_kendall_tau"].mean(),
            }
        )
    return pd.DataFrame(rows)


def analyse_lineage(adata, lineage, output_dir):
    """Filter to the lineage and compute stage-order, gene-trend and module-trend metrics (script block 1)."""
    TARGET_LINEAGE_NAME = lineage
    ACTIVE_CONFIG = LINEAGE_CONFIGS[lineage]
    TARGET_CELLTYPES = ACTIVE_CONFIG["target_celltypes"]
    STAGE_RANK = ACTIVE_CONFIG["stage_rank"]
    REFERENCE_MODULES = ACTIVE_CONFIG["reference_modules"]
    EXPECTED_GENE_DIRECTION = ACTIVE_CONFIG["expected_gene_direction"]
    REPRESENTATIVE_GENES_FOR_PLOT = ACTIVE_CONFIG["representative_genes_for_plot"]
    OUTPUT_DIR = str(output_dir)
    #%% filter adata to target lineage and check pseudotime validity
    check_normalization(adata)

    adata = adata[adata.obs[CELLTYPE_COL].isin(TARGET_CELLTYPES)].copy()
    if adata.n_obs == 0:
        raise ValueError("No cells remain after lineage subsetting. Check TARGET_CELLTYPES.")

    print(f"[INFO] {TARGET_LINEAGE_NAME}: retained {adata.n_obs} cells after cell-type filtering.")

    valid_mask = ~adata.obs[PSEUDOTIME_COL].isna().to_numpy()
    removed = int((~valid_mask).sum())
    if removed > 0:
        print(f"[INFO] Removed {removed} cells with NaN pseudotime.")
    adata = adata[valid_mask].copy()

    if adata.n_obs < 20:
        raise ValueError(f"Too few cells remain after pseudotime filtering: {adata.n_obs}")

    print(f"[INFO] Proceeding with {adata.n_obs} cells for trajectory validation.")

    #%%
    stage_df, stage_summary_df = compute_stage_order_metrics(
        adata=adata,
        celltype_col=CELLTYPE_COL,
        pseudotime_col=PSEUDOTIME_COL,
        stage_rank=STAGE_RANK,
    )
    #%%
    gene_metric_df = compute_gene_trend_metrics(
        adata=adata,
        pseudotime_col=PSEUDOTIME_COL,
        expected_direction=EXPECTED_GENE_DIRECTION,
    )
    module_metric_df = summarize_module_trend_metrics(gene_metric_df, REFERENCE_MODULES)

    # outputs = {
    #     "stage_order_table.csv": stage_df,
    #     "stage_order_metrics.csv": stage_summary_df,
    #     "gene_trend_metrics.csv": gene_metric_df,
    #     "module_trend_metrics.csv": module_metric_df,
    # }
    # for filename, df in outputs.items():
    #     outpath = os.path.join(OUTPUT_DIR, filename)
    #     df.to_csv(outpath, index=False)
    #     print(f"[INFO] Saved {outpath}")

    if not stage_df.empty:
        print("\n[INFO] Stage-order table:")
        print(stage_df.to_string(index=False))

    print("\n[INFO] Stage-order summary:")
    print(stage_summary_df.to_string(index=False))

    print("\n[INFO] Module trend summary:")
    print(module_metric_df.to_string(index=False))

    print(f"\n[INFO] Analysis complete for {TARGET_LINEAGE_NAME}. Results saved to {os.path.basename(os.path.normpath(str(OUTPUT_DIR)))}/")

    return adata, stage_df, stage_summary_df, gene_metric_df, module_metric_df


def plot_gene_trends(adata, lineage, output_dir):
    """<gene>_<lineage>_pseudotime_trend.pdf for the representative genes (script block 2)."""
    TARGET_LINEAGE_NAME = lineage
    ACTIVE_CONFIG = LINEAGE_CONFIGS[lineage]
    TARGET_CELLTYPES = ACTIVE_CONFIG["target_celltypes"]
    STAGE_RANK = ACTIVE_CONFIG["stage_rank"]
    REFERENCE_MODULES = ACTIVE_CONFIG["reference_modules"]
    EXPECTED_GENE_DIRECTION = ACTIVE_CONFIG["expected_gene_direction"]
    REPRESENTATIVE_GENES_FOR_PLOT = ACTIVE_CONFIG["representative_genes_for_plot"]
    OUTPUT_DIR = str(output_dir)
    # %% Draw the gene expression of the representative genes along pseudotime for visualization
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import scanpy as sc
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    # Publication-style settings
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams['figure.figsize'] = (8, 3)

    sc.settings.set_figure_params(
        dpi=600,
        facecolor="white",
        fontsize=14,
        vector_friendly=True,
    )


    for gene in REPRESENTATIVE_GENES_FOR_PLOT:
        if gene not in adata.var_names:
            print(f"[WARNING] Representative gene '{gene}' not found in adata.var_names; skipping plot.")
            continue
        fig, ax = plt.subplots(figsize=(8, 3))
        sc.pl.scatter(
            adata,
            x=PSEUDOTIME_COL,
            y=gene,
            color=CELLTYPE_COL,
            title=f"",
            size=20,
            alpha=0.8,
            show=False,
            ax=ax,
        )

        ax.set_xlabel("Pseudotime", fontsize=16)
        ax.set_ylabel(f"{gene} expression", fontsize=16)
        ax.grid(False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))  # 控制刻度数量
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("Cell Type", prop={"size": 15})
            for text in legend.get_texts():
                text.set_fontsize(13)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTPUT_DIR, f"{gene}_{TARGET_LINEAGE_NAME}_pseudotime_trend.pdf"
                ), 
                bbox_inches="tight",
                format="pdf",
            )
        plt.close(fig)
        # plt.show()


def plot_module_radar(module_metric_df, lineage, output_dir):
    """<lineage>_module_trend_summary.pdf (script block 3)."""
    TARGET_LINEAGE_NAME = lineage
    ACTIVE_CONFIG = LINEAGE_CONFIGS[lineage]
    TARGET_CELLTYPES = ACTIVE_CONFIG["target_celltypes"]
    STAGE_RANK = ACTIVE_CONFIG["stage_rank"]
    REFERENCE_MODULES = ACTIVE_CONFIG["reference_modules"]
    EXPECTED_GENE_DIRECTION = ACTIVE_CONFIG["expected_gene_direction"]
    REPRESENTATIVE_GENES_FOR_PLOT = ACTIVE_CONFIG["representative_genes_for_plot"]
    OUTPUT_DIR = str(output_dir)
    #%%
    import numpy as np
    import matplotlib.pyplot as plt


    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300
    # mpl.rcParams['figure.figsize'] = (8, 3)


    metrics = [
        "median_signed_spearman_rho",
        "median_signed_kendall_tau",
        "mean_signed_spearman_rho",
        "mean_signed_kendall_tau",
    ]

    metric_labels = [
        "Median\nSpearman",
        "Median\nKendall",
        "Mean\nSpearman",
        "Mean\nKendall",
    ]

    # 你原来选的数据颜色
    module_colors = ["#7AA6C2", "#F2A65A", "#E76F51", "#9D79BC"]

    df = module_metric_df.copy()

    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FCFCFD")

    # 从正上方开始，顺时针
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 画每个 module
    for i, (_, row) in enumerate(df.iterrows()):
        values = row[metrics].tolist()
        values += values[:1]

        color = module_colors[i % len(module_colors)]
        module_name = row["module"].replace("_", " ").title()

        ax.plot(
            angles,
            values,
            color=color,
            linewidth=2.8,
            marker="o",
            markersize=6,
            label=f"{module_name} (n={row['n_genes_present']})"
        )
        ax.fill(angles, values, color=color, alpha=0.14)

    # 不用默认 xtick 文本，手动放标签，方便调整左右位置
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    # 指标名称：统一黑色；左右两个标签稍微往外放
    for idx, (angle, label) in enumerate(zip(angles[:-1], metric_labels)):
        r = 1.10
        if idx in [1, 3]:  # 右边和左边的两个标签
            r = 1.20
        ax.text(
            angle,
            r,
            label,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color="black"
        )

    # 半径范围和标尺
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=13, color="black")
    ax.tick_params(axis="y", pad=12)

    # 网格更清楚一点
    ax.yaxis.grid(True, linestyle="--", linewidth=0.9, alpha=0.45)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.7, alpha=0.25)

    # 标题
    ax.set_title(
        "",
        fontsize=17,
        fontweight="bold",
        pad=28,
        color="black"
    )

    # legend 不要框
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.95, 0.99),
        frameon=False,
        fontsize=16
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"{TARGET_LINEAGE_NAME}_module_trend_summary.pdf"),
        bbox_inches="tight",
        format="pdf",
    )
    plt.show()


