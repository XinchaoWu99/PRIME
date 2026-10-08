"""scGPT zero-shot benchmark, step 4: put the scGPT row next to the benchmark table of the other methods.

    python collect_tables.py DATASET

Reads the saved benchmark table (config.yaml benchmark.<dataset>.results_csv) and every
out_dir/scgpt/<dataset>/scores/*.csv written by score_embeddings.py, and writes into the same scores/ folder:
  benchmark_results_with_scgpt.csv   the saved rows unchanged + one scGPT row, "scGPT (zero-shot)" (same layout, same
                                     aggregate scores); this is the table that is drawn
  scgpt_runs.csv                     every scored scGPT run (one row per run: zero-shot on the HVGs, zero-shot on all genes)
Scored embeddings are named scGPT_zeroshot_hvg and scGPT_zeroshot_all; the re-scored reference methods (reproduction
check) are not merged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import common as C

LABELS = {"scGPT_zeroshot_hvg": "scGPT (zero-shot)"}
METRIC_TYPE = "Metric Type"


def main(dataset: str) -> None:
    cfg = C.load_config()
    d = Path(cfg["out_dir"]) / "scgpt" / dataset / "scores"
    man = pd.read_csv(cfg["benchmark"][dataset]["results_csv"], index_col=0)
    runs = []
    for f in sorted(d.glob("*.csv")):
        if f.name.endswith("_vs_benchmark.csv") or f.name.startswith(("benchmark_results_with_scgpt", "scgpt_")):
            continue
        df = pd.read_csv(f, index_col=0)
        runs.append(df.drop(index=METRIC_TYPE)[[r.startswith("scGPT_") for r in df.drop(index=METRIC_TYPE).index]])
    runs = pd.concat(runs)
    runs = runs[~runs.index.duplicated(keep="last")].astype(float)
    assert list(runs.columns) == [c for c in man.columns], (list(runs.columns), list(man.columns))
    runs.index.name = "Embedding"
    runs.to_csv(d / "scgpt_runs.csv")

    rows = {}
    for k, label in LABELS.items():
        if k in runs.index:
            rows[label] = runs.loc[k]
    new = pd.DataFrame(rows).T
    new.index.name = "Embedding"
    metric_row = man.loc[[METRIC_TYPE]]
    merged = pd.concat([man.drop(index=METRIC_TYPE), new.astype(float), metric_row])
    merged.index.name = "Embedding"
    merged.to_csv(d / "benchmark_results_with_scgpt.csv")
    print(merged.drop(index=METRIC_TYPE).astype(float).round(3).sort_values("Total", ascending=False).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
