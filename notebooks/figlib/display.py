"""Small previews in the notebooks (the full-resolution files are written to the figure folder)."""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import Image, display
from PIL import Image as PImage


def rel(path) -> str:
    """`path` relative to the repository root when it lies inside it (for messages)."""
    root = Path(__file__).resolve().parents[2]
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return str(path)


def show_png(*paths, max_px: int = 700) -> None:
    """Display down-scaled copies of PNG files (keeps the executed notebooks small)."""
    for p in paths:
        im = PImage.open(p)
        im.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        print(Path(p).name)
        display(Image(data=buf.getvalue()))


def preview(fig, dpi: int = 60) -> None:
    """Show an already saved matplotlib figure at low resolution, then close it."""
    fig.set_dpi(dpi)
    display(fig)
    plt.close(fig)
