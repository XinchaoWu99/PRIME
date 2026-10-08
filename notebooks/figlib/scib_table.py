"""scIB-style results table (plottable) with the isolated-label column, used for the benchmark tables of the human
immune and NSCLC data (the table function of pipeline/benchmark/nsclc/02_run_benchmark.py).
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.text import Text
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar

RESULTS_METRIC_TYPE_ROW = "Metric Type"
AGGREGATE_SCORE_VALUE = "Aggregate score"
BIO_CONSERVATION_COL = "Bio conservation"
BATCH_CORRECTION_COL = "Batch correction"
TOTAL_SCORE_COL = "Total"
ISOLATED_LABELS_COL = "Isolated labels"


def _validate_results_frame(df: pd.DataFrame) -> None:
    if RESULTS_METRIC_TYPE_ROW not in df.index:
        raise ValueError(
            "Input DataFrame must include a 'Metric Type' row, as returned by "
            "Benchmarker.get_results()."
        )


def _resolve_score_columns(df: pd.DataFrame) -> list[str]:
    metric_type_row = df.loc[RESULTS_METRIC_TYPE_ROW]
    score_cols = metric_type_row.index[metric_type_row == AGGREGATE_SCORE_VALUE].tolist()
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

    for candidate in ("Total", "Batch correction", "Bio conservation"):
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

    plot_df = results_df.drop(index=RESULTS_METRIC_TYPE_ROW)
    plot_df = plot_df.sort_values(by=resolved_sort_col, ascending=False).astype(np.float64)
    plot_df["Method"] = plot_df.index

    metric_type_row = results_df.loc[RESULTS_METRIC_TYPE_ROW]
    other_cols = metric_type_row.index[metric_type_row != AGGREGATE_SCORE_VALUE].tolist()
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
    font_family: str = "Arial",
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.02,
) -> None:
    """Save an existing figure as a publication-ready PDF with editable text."""

    pdf_path = Path(output_path)
    if pdf_path.suffix.lower() != ".pdf":
        pdf_path = pdf_path.with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    _set_figure_text_font_family(fig, font_family)
    with mpl.rc_context(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
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
