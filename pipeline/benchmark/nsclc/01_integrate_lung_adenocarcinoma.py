"""NSCLC atlas, lung adenocarcinoma: subset, preprocessing and the Python integration methods (step 1).

Reads the NSCLC atlas (Salcher et al. 2022; config.yaml `datasets.lung.path`), keeps the lung-adenocarcinoma cells
(410,927 cells of 12 datasets; batch = dataset, cell_type = ann_fine), normalises (CP10k, log1p), selects 2,000
batch-aware HVGs, and runs Scanorama, Harmony, MNN, ComBat, BBKNN, scVI and PRIME. Writes
`benchmarked_lung_cancer_integrated.h5ad` (HVG matrix, every embedding in obsm) into config.yaml
`benchmark.lung.dir`.

Run order: 01 (this) -> ../r_methods/run_r_methods.R -> 02 (benchmark); robustness: 03 -> 04 -> 05 -> 06.
"""
# %%
import os
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # pipeline/ (common.py)
import common as C  # noqa: E402

CFG = C.load_config()

nsclc_atlas_file = CFG["datasets"]["lung"]["path"]
out_dir = CFG["benchmark"]["lung"]["dir"]
nsclc_atlas = sc.read_h5ad(nsclc_atlas_file, backed="r")
nsclc_atlas

adata = nsclc_atlas[nsclc_atlas.obs["disease"] == "lung adenocarcinoma"].to_memory()

del nsclc_atlas
gc.collect()

adata.obs['batch'] = adata.obs['dataset']
adata.obs["cell_type"] = adata.obs["ann_fine"]
adata.X = adata.layers["count"].copy()
# %%
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key="batch")
sc.tl.pca(adata, n_comps=30, use_highly_variable=True)
adata = adata[:, adata.var.highly_variable].copy()
adata.obsm["Unintegrated"] = adata.obsm["X_pca"]
# %%
# scanorama integration
import scanorama

# List of adata per batch
batch_cats = adata.obs.batch.cat.categories
adata_list = [adata[adata.obs.batch == b].copy() for b in batch_cats]
scanorama.integrate_scanpy(adata_list, dimred=100, hvg=2000)

adata.obsm["Scanorama"] = np.zeros((adata.shape[0], adata_list[0].obsm["X_scanorama"].shape[1]))
for i, b in enumerate(batch_cats):
    adata.obsm["Scanorama"][adata.obs.batch == b] = adata_list[i].obsm["X_scanorama"]


# harmony integration
from harmonypy import run_harmony

harmony_out = run_harmony(adata.obsm["X_pca"], adata.obs, "batch")
adata.obsm["Harmony"] = harmony_out.Z_corr

# mnn integration
batch_cats = adata.obs.batch.cat.categories
adata_list = [adata[adata.obs.batch == b].copy() for b in batch_cats]
hvgs = adata.var[adata.var.highly_variable].index.tolist()

corrected = sc.external.pp.mnn_correct(
    adata_list,
    var_subset=hvgs,
    batch_key='batch',
    # batch_categories=['batch1', 'batch2', 'batch3'],
    k=20,
    sigma=0.1,
    svd_dim=100,  # Number of SVD dimensions
    save_raw=True,  # Save uncorrected data in .raw
    do_concatenate=True,  # Return concatenated object
    n_jobs=16,
)

adata_corrected = ad.concat(corrected[0][0])
adata_corrected = adata_corrected[adata.obs_names, :]
adata.obsm["MNN"] = adata_corrected.X.toarray()

# combat integration
adata_combat_expression = adata.copy()
adata_combat_expression = sc.pp.combat(adata, key="batch", inplace=False)

adata_combat = ad.AnnData(X=adata_combat_expression)
adata_combat.obs = adata.obs.copy()
adata_combat.var = adata.var.copy()

adata.obsm["ComBat"] = adata_combat.X

# bbknn integration
import bbknn

adata_highly_var = adata[:, adata.var.highly_variable].copy()
neighbors_within_batch = 3
adata_bbknn = adata_highly_var.copy()
adata_highly_var.layers["logcounts"] = adata_highly_var.X.copy()
adata_bbknn.X = adata_highly_var.layers["logcounts"].copy()
sc.pp.pca(adata_bbknn)
bbknn.bbknn(
    adata_bbknn, batch_key="batch", neighbors_within_batch=neighbors_within_batch
)
adata.obsm["BBKNN"] = adata_bbknn.X.toarray().copy()

# scVI
import scvi

adata_scvi = adata_highly_var.copy()
scvi.model.SCVI.setup_anndata(adata_scvi, layer="count", batch_key="batch")
model_scvi = scvi.model.SCVI(adata_scvi)
max_epochs_scvi = np.min([round((20000 / adata_scvi.n_obs) * 400), 400])
model_scvi.train()
adata_scvi.obsm["X_scVI"] = model_scvi.get_latent_representation()
adata.obsm["scVI"] = adata_scvi.obsm["X_scVI"].copy()

# PRIME
from prime import ensemble_mnn_correct

adata_highly_var.obsm["PRIME"] = ensemble_mnn_correct(
    adata_highly_var,
    batch_key="batch",
    n_projections=4,
    target_dim=128,
    k_neighbors=15,
    consensus_threshold=0.4,
    sigma=0.1,
    random_state=42,
    key_added=None,
    inplace=False
)
adata.obsm["PRIME"] = adata_highly_var.obsm["PRIME"].copy()

os.makedirs(out_dir, exist_ok=True)
save_file = f"{out_dir}/benchmarked_lung_cancer_integrated.h5ad"
adata.write(save_file)
