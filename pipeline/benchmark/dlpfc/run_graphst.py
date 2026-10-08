"""GraphST on the 12 DLPFC sections (merged object, 2,000 HVGs, dim_output 32).

    python run_graphst.py

Reads DLPFC_merged.h5ad from config.yaml `benchmark.dlpfc.data_dir` and writes GraphST_embedding.npy (one row per spot,
file order) into `benchmark.dlpfc.results_dir`.
"""
import os
import sys
from pathlib import Path

n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 16))   # CPU cores available to the job

os.environ["OMP_NUM_THREADS"] = str(n_threads)
os.environ["MKL_NUM_THREADS"] = str(n_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

import torch

torch.set_num_threads(n_threads)
torch.set_num_interop_threads(1)

from GraphST import GraphST
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
print("PyTorch threads:", torch.get_num_threads())

import numpy as np
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # pipeline/ (common.py)
import common as C  # noqa: E402

CFG = C.load_config()

DLFPC_data_dir = CFG["benchmark"]["dlpfc"]["data_dir"]
DLFPC_merged_data_file = os.path.join(DLFPC_data_dir, "DLPFC_merged.h5ad")

DLFPC_adata = sc.read_h5ad(DLFPC_merged_data_file)
DLFPC_adata

adata = DLFPC_adata.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=2000,
    flavor="seurat_v3"
)

adata = adata[:, adata.var["highly_variable"]].copy()

model = GraphST.GraphST(adata, 
            dim_output=32,
            device=device
            )
adata = model.train()

integrated_embeddings = adata.obsm["emb"].copy()

emb_save_dir = CFG["benchmark"]["dlpfc"]["results_dir"]
emb_save_file = os.path.join(emb_save_dir, "GraphST_embedding.npy")

if not os.path.exists(emb_save_dir):
    os.makedirs(emb_save_dir)
np.save(emb_save_file, integrated_embeddings)
