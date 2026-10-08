"""scGPT zero-shot benchmark, step 2: cell embeddings from the pre-trained scGPT (whole-human) model.

    python embed_cells.py DATASET [--genes hvg|all] [--seed 0] [--subsample N]

Runs in an environment with scGPT (0.2.4, torch 2.1) on a GPU; reads the files of prepare_inputs.py.

`scgpt.tasks.embed_data` of the zero-shot integration tutorial (tutorials/zero-shot/Tutorial_ZeroShot_Integration):
the pre-trained model is applied as it is, cell embedding = <cls> output, no batch information, no training.
Input = log-normalised expression of the 2,000 HVGs used by every other method (`hvg`, the reported run) or of all
genes (`all`, sensitivity run).

Outputs: out_dir/scgpt/<dataset>/embeddings/scgpt_zeroshot_<genes>_s<seed>.npy (cells x 512, rows in the order of
obs.tsv, L2-normalised as in the tutorial) and the matching .json (timings, matched genes).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pipeline/ (common.py, anchors.py)
import common as C  # noqa: E402

warnings.filterwarnings("ignore")
CFG = C.load_config()
SCGPT = CFG["scgpt"]
MODEL_DIR = Path(SCGPT["model_dir"])
PAD, CLS, EOC = "<pad>", "<cls>", "<eoc>"


def load_inputs(name: str):
    d = Path(CFG["out_dir"]) / "scgpt" / name
    return d, pd.read_csv(d / "obs.tsv", sep="\t", index_col=0), pd.read_csv(d / "genes.tsv", sep="\t")


def pick_cells(n_obs: int, subsample: int | None, seed: int) -> np.ndarray:
    return np.arange(n_obs) if not subsample else np.sort(np.random.default_rng(seed).choice(n_obs, subsample, replace=False))


def make_vocab():
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    vocab = GeneVocab.from_file(MODEL_DIR / "vocab.json")
    for s in (PAD, CLS, EOC):
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab[PAD])
    return vocab


def match_genes(symbols: pd.Series, vocab) -> np.ndarray:
    """Boolean mask of the genes kept for scGPT: symbol in the vocabulary, first occurrence of a symbol only."""
    in_vocab = symbols.map(lambda g: g in vocab).to_numpy()
    first = ~symbols.duplicated().to_numpy()
    return in_vocab & first


def lognorm(counts: sp.csr_matrix) -> sp.csr_matrix:
    x = counts.astype(np.float32).tocsr()
    size = np.asarray(x.sum(1)).ravel()
    x = sp.diags(1e4 / np.maximum(size, 1)) @ x
    x.data = np.log1p(x.data)
    return x.tocsr().astype(np.float32)


def zeroshot(name: str, genes_set: str, seed: int, subsample: int | None) -> None:
    import anndata
    import scgpt as scg
    from scgpt.utils import set_seed
    set_seed(seed)
    d, obs, genes = load_inputs(name)
    t0 = time.time()
    if genes_set == "hvg":
        X = sp.load_npz(d / "lognorm_hvg.npz").tocsr()
        genes = genes[genes["is_hvg"]].reset_index(drop=True)
    else:
        X = lognorm(sp.load_npz(d / "counts.npz").tocsr())
    vocab = make_vocab()
    keep = match_genes(genes["symbol"], vocab)
    print(f"{name}/{genes_set}: {keep.sum()}/{len(genes)} genes usable (in vocabulary, unique symbol)", flush=True)
    cells = pick_cells(X.shape[0], subsample, seed)
    X = X[cells][:, np.flatnonzero(keep)]
    var = pd.DataFrame({"gene_name": genes["symbol"].values[keep]}, index=genes["var_id"].values[keep])
    emb = np.zeros((len(cells), 512), dtype=np.float32)
    zs = SCGPT["zeroshot"]
    chunk = 50000
    for a in range(0, len(cells), chunk):
        ad = anndata.AnnData(X=X[a:a + chunk].astype(np.float32), obs=pd.DataFrame(index=obs.index.values[cells[a:a + chunk]]), var=var)
        out = scg.tasks.embed_data(ad, MODEL_DIR, gene_col="gene_name", max_length=zs["max_length"],
                                   batch_size=zs["batch_size"], device="cuda", use_fast_transformer=False)
        emb[a:a + chunk] = out.obsm["X_scGPT"]
        print(f"  embedded {min(a + chunk, len(cells))}/{len(cells)} cells ({time.time() - t0:.0f} s)", flush=True)
    save(d, f"zeroshot_{genes_set}", seed, subsample, emb, cells, {
        "genes_used": int(keep.sum()), "genes_total": int(len(genes)), "seconds": time.time() - t0,
        "gpu": torch.cuda.get_device_name(0), **zs})


def save(d: Path, variant: str, seed: int, subsample: int | None, emb: np.ndarray, cells: np.ndarray, meta: dict) -> None:
    assert np.isfinite(emb).all(), "non-finite values in the embedding"
    out = d / ("embeddings" if not subsample else "embeddings_subsample")
    out.mkdir(exist_ok=True)
    stem = out / f"scgpt_{variant}_s{seed}"
    np.save(f"{stem}.npy", emb)
    np.save(f"{stem}_cells.npy", cells)
    json.dump({"variant": variant, "seed": seed, "n_cells": int(len(cells)), **meta}, open(f"{stem}.json", "w"), indent=1)
    print("saved", stem, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["immune", "lung"])
    ap.add_argument("--genes", default="hvg", choices=["hvg", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subsample", type=int, help="use only this many random cells (writes to embeddings_subsample/)")
    a = ap.parse_args()
    zeroshot(a.dataset, a.genes, a.seed, a.subsample)
