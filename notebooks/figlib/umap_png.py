"""Point-only UMAP PNGs, one per colour key (UMAP panels of the human immune and DLPFC data).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc

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
