"""Method names in saved results.

Result files written by earlier versions of the pipeline store PRIME under the name of its core step, "ERP_MNN"
(ensemble random projection + mutual nearest neighbours). The helpers below rename it when such files are read, so
that the notebooks work with results of either naming.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LEGACY_NAMES = {"ERP_MNN": "PRIME", "erp_mnn": "prime"}


def canonical(name: str) -> str:
    return LEGACY_NAMES.get(name, name)


def rename_obsm(adata) -> None:
    """Rename legacy embedding keys of `adata.obsm` in place."""
    for old, new in LEGACY_NAMES.items():
        if old in adata.obsm and new not in adata.obsm:
            adata.obsm[new] = adata.obsm[old]
            del adata.obsm[old]


def rename_methods(df: pd.DataFrame, columns=("method", "Method")) -> pd.DataFrame:
    """Legacy method names in the index and in the given columns replaced by the current ones."""
    df = df.rename(index=LEGACY_NAMES)
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(LEGACY_NAMES)
    return df


def method_file(pattern: str, method: str) -> Path:
    """`pattern.format(method=...)`, falling back to a legacy method name when only that file exists."""
    path = Path(pattern.format(method=method))
    if not path.exists():
        for old, new in LEGACY_NAMES.items():
            alt = Path(pattern.format(method=old))
            if new == method and alt.exists():
                return alt
    return path
