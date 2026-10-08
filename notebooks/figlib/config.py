"""Locations used by the figure notebooks: the pipeline configuration (pipeline/config.yaml, with
pipeline/config.local.yaml merged over it) plus the figure and cache folders."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]              # repository root
sys.path.insert(0, str(ROOT / "pipeline"))              # common.py and the analysis scripts
import common as C  # noqa: E402


def load() -> dict:
    """The pipeline configuration; `figure_dir`, `cache_dir`, `data_root` and `out_dir` as absolute paths."""
    cfg = C.load_config()
    for key in ("figure_dir", "cache_dir"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else ROOT / p
    for key in ("data_root", "out_dir"):
        cfg[key] = Path(cfg[key])
    return cfg


def figure_dir(*parts: str) -> Path:
    """figures/<parts...>, created if needed."""
    d = load()["figure_dir"].joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir(*parts: str) -> Path:
    d = load()["cache_dir"].joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
