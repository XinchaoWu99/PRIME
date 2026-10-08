"""Setting selection for every spatial method, using the development donor only.

    python select_settings.py

For every method in config.yaml `baselines` and every setting with a seed-0 embedding
(out_dir/spatial_methods/<method>/<setting>_s0.npy, written by run_methods.py), the score is computed on the
sections of the development donor only:
    mean( per-section ARI of GMM domains (K = 7, 3 clustering seeds),
          leave-one-section-out kNN layer accuracy across that donor's sections )
Both parts need only one donor, so jointly and per-donor integrated methods are scored the same way.
Writes out_dir/spatial_methods/tuning_log.tsv and tuning_selected.tsv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import common as C

BASE = {"spacross": "donor"}          # SpaCross cannot run on all 12 sections at once; its settings vary the per-donor one
SKIP = {"spabatch": {"donor"}, "spacross": {"default"}, "prime": {"unintegrated", "transport"}, "staligner": {"hd16"}}


def dev_score(Z, lab, sl, seeds=(0, 1, 2)):
    ok = np.isfinite(Z).all(1) & (lab != "nan")
    Z, lab, sl = C.eval_view(Z[ok], 20), lab[ok], sl[ok]
    ari = np.mean([np.mean([adjusted_rand_score(lab[sl == s], d[sl == s]) for s in np.unique(sl)])
                   for d in (GaussianMixture(7, covariance_type="full", reg_covar=1e-4, random_state=r).fit_predict(Z)
                             for r in seeds)])
    acc = np.mean([(KNeighborsClassifier(15).fit(Z[sl != s], lab[sl != s]).predict(Z[sl == s]) == lab[sl == s]).mean()
                   for s in np.unique(sl)])
    return ari, acc


def main():
    cfg = C.load_config()
    out = Path(cfg["out_dir"]) / "spatial_methods"
    adata = C.load_dataset("dlpfc", cfg)
    dev = (adata.obs["donor"] == cfg["datasets"]["dlpfc"]["dev_donor"]).to_numpy()
    lab = adata.obs["label"].astype(str).to_numpy()[dev]
    sl = adata.obs["batch"].astype(str).to_numpy()[dev]

    rows = []
    for method, settings in cfg["baselines"].items():
        for name in settings:
            f = out / method / f"{name}_s0.npy"
            if name in SKIP.get(method, ()) or not f.exists():
                continue
            ari, acc = dev_score(np.load(f)[dev], lab, sl)
            rows.append({"method": method, "setting": name, "params": str(settings[name]),
                         "dev_ARI": ari, "dev_transfer_acc": acc, "score": (ari + acc) / 2})
            print(rows[-1], flush=True)
    log = pd.DataFrame(rows)
    log.to_csv(out / "tuning_log.tsv", sep="\t", index=False)
    best = log.loc[log.groupby("method")["score"].idxmax()].copy()
    best["base_setting"] = best["method"].map(lambda m: BASE.get(m, "default"))
    best["base_score"] = [log[(log.method == m) & (log.setting == b)]["score"].max()
                          for m, b in zip(best.method, best.base_setting)]
    best.to_csv(out / "tuning_selected.tsv", sep="\t", index=False)
    print(best[["method", "setting", "score", "base_setting", "base_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
