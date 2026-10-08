"""DLPFC (12 Visium sections): scIB metrics, isolated-label score and XLC of the spatial integration methods.

Cell-structured script (`# %%` cells). Reads DLPFC_merged.h5ad (config.yaml `benchmark.dlpfc.data_dir`) and, from
`benchmark.dlpfc.results_dir`, one embedding per method with one row per spot in file order
(<method>_embedding.npy; PRECAST: PRECAST_embedding_with_meta.csv of run_precast.R, whose spots removed by PRECAST's
own filtering receive the mean embedding). Unintegrated reference = PCA (50) of the scaled 3,000 HVGs.

  scib-metrics Benchmarker (KMeans NMI / ARI, silhouette label, cLISI, BRAS, iLISI, kBET), isolated-label silhouette,
  XLC (`prime.metrics.xlc_score`: ordinal continuity of the cortical layers across sections).

Writes benchmark_results_with_xlc.csv and benchmark_results_table.pdf into the results folder.
"""
#%%
from __future__ import annotations
 
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union

import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # pipeline/ (common.py)
import common as C  # noqa: E402

CFG = C.load_config()
 
# %%
DLFPC_data_dir = CFG["benchmark"]["dlpfc"]["data_dir"]
DLFPC_merged_data_file = os.path.join(DLFPC_data_dir, "DLPFC_merged.h5ad")

DLFPC_adata = sc.read_h5ad(DLFPC_merged_data_file)
DLFPC_adata

adata = DLFPC_adata.copy()
adata.obs['section'] = adata.obs['sample'].copy()

# %% run pca
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3')
adata_tmp = adata[:, adata.var['highly_variable']].copy()
sc.pp.scale(adata_tmp, max_value=10)
sc.tl.pca(adata_tmp, n_comps=50, svd_solver='arpack')
adata.obsm['Unintegrated'] = adata_tmp.obsm['X_pca']

# %%
save_dir = CFG["benchmark"]["dlpfc"]["results_dir"]
precast_df = pd.read_csv(f"{save_dir}/PRECAST_embedding_with_meta.csv")

precast_df['key'] = precast_df['section'].astype(str) + '_' + precast_df['barcode'].astype(str)
embedding_cols = [f"PRECAST_{i}" for i in range(1, 16)]
precast_df = precast_df.set_index('key')


all_keys = adata.obs['key'].values
missing_keys = set(all_keys) - set(precast_df.index)
print(f"spots without a PRECAST embedding: {len(missing_keys)}")

# mean of every dimension = the embedding given to the spots PRECAST did not return
mean_embedding = precast_df[embedding_cols].mean(axis=0).values  # shape: (15,)

# one row per missing spot
for mk in missing_keys:
    precast_df.loc[mk, embedding_cols] = mean_embedding
    section_val = adata.obs.loc[adata.obs['key'] == mk, 'section'].iloc[0]
    barcode_val = adata.obs.loc[adata.obs['key'] == mk, 'barcode'].iloc[0]
    precast_df.loc[mk, 'section'] = section_val
    precast_df.loc[mk, 'barcode'] = barcode_val

# reorder the embedding matrix to the spot order of adata.obs['key']
embedding_matrix = precast_df.loc[all_keys, embedding_cols].values  # shape: (N, 15)

assert embedding_matrix.shape == (adata.n_obs, 15), embedding_matrix.shape

adata.obsm['X_precast'] = embedding_matrix

print(f"adata.obsm['X_precast'] shape: {adata.obsm['X_precast'].shape}")
# %% loading embeddings
results_dir = CFG["benchmark"]["dlpfc"]["results_dir"]
methods_list = [
    "STAligner", "paste", "GraphST", "prime"
]

for method in methods_list:
    embedding_file = os.path.join(results_dir, f"{method}_embedding.npy")
    if os.path.exists(embedding_file):
        embedding = np.load(embedding_file)
        adata.obsm[f"X_{method}"] = embedding
        print(f"Loaded embedding for {method} from {embedding_file}")
    else:
        print(f"Embedding file for {method} not found at {embedding_file}")

# %%
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

benchmark_keys = ["STAligner", "paste", "GraphST", "prime", "precast"]
benchmark_keys = [f"X_{method}" for method in benchmark_keys] + ["Unintegrated"]

adata = adata[adata.obs['layer_guess'].notna()].copy()
bm = Benchmarker(
    adata,
    batch_key="sample",
    label_key="layer_guess",
    bio_conservation_metrics=BioConservation(
            nmi_ari_cluster_labels_leiden=False, 
            nmi_ari_cluster_labels_kmeans=True, 
            silhouette_label=True,
            isolated_labels=False,
        ),
    batch_correction_metrics=BatchCorrection(
            graph_connectivity=False,
            pcr_comparison=False,
        ),
    embedding_obsm_keys=benchmark_keys,
    n_jobs=16,
)
bm.benchmark()

#%% plot the results and save the results table
bm.plot_results_table()

results_df = bm.get_results()
results_df
# results_df.to_csv(f"{results_dir}/benchmark_results.csv", index=True)

# %% calculate the isolated labels metric
from prime.metrics.isolated_label import compute_isolated_label_scores, add_isolated_label_scores_to_results

isolated_results_df = compute_isolated_label_scores(
    adata,
    label_key="layer_guess",
    embedding_keys=benchmark_keys,
    batch_key="sample",
    rescale=True,
    iso_threshold=None,
    verbose=True,
)
results_df = add_isolated_label_scores_to_results(results_df, isolated_results_df)
results_df

# results_df.to_csv(revised_res_df_file, index=True)
# %% compute xlc scores
from prime.metrics import xlc_score

xlc_results = xlc_score(
    adata,
    label_key="layer_guess",
    embedding_keys=benchmark_keys,
    batch_key="sample",
    obs_only=True,
)

# add one row for metric type
xlc_results.loc["Metric Type"] = "Spatial Continuity"

# append xlc results to the existing results_df
results_df["XLC"] = xlc_results
results_df
# %% update the aggregate scores and put the isolated label scores as the second column
update_col = (
    0.7 * results_df.loc[benchmark_keys]['Bio conservation']
    + 0.15 * results_df.loc[benchmark_keys]['Batch correction']
    + 0.15 * results_df.loc[benchmark_keys]['XLC']
)

update_col["Metric Type"] = "Aggregate score"

results_df["Total Score"] = update_col

# Sort
# results_df = results_df.sort_values('Total Score', ascending=False)

results_df
# %%
results_df.to_csv(f"{results_dir}/benchmark_results_with_xlc.csv", index=True)
# %%
import pandas as pd
import matplotlib.pyplot as plt

results_dir = CFG["benchmark"]["dlpfc"]["results_dir"]
results_df = pd.read_csv(f"{results_dir}/benchmark_results_with_xlc.csv", index_col=0)

benchmark_keys = ["STAligner", "paste", "GraphST", "prime", "precast"]
benchmark_keys = [f"X_{method}" for method in benchmark_keys] + ["Unintegrated"]
results_df = results_df.loc[benchmark_keys + ["Metric Type"]].copy()      # the methods of the table

# use Total Score to replace Total
results_df.drop(columns=["Total"], inplace=True)
results_df.rename(columns={"Total Score": "Total"}, inplace=True)

# drop out the KBET and re-calculate the aggregate score
results_df.drop(columns=["KBET"], inplace=True)
BatchCorrection_cols = ["BRAS"]

update_col = results_df.loc[benchmark_keys, BatchCorrection_cols].copy()
update_col = update_col.apply(pd.to_numeric, errors='coerce')

update_col["Batch correction"] = update_col.mean(axis=1)

results_df.loc[benchmark_keys, "Batch correction"] = update_col["Batch correction"]

#  re-calculate the Total score
update_col = results_df.loc[benchmark_keys].copy()
update_col = update_col.apply(pd.to_numeric, errors='coerce')

results_df.loc[benchmark_keys, "Total"] = (
    0.7 * update_col['Bio conservation']
    + 0.05 * update_col['Batch correction']
    + 0.25 * update_col['XLC']
)

results_df.loc["Metric Type", "Total"] = "Aggregate score"

# rename the display names for the methods: "X_method" -> "method"
# "prime" --> "PRIME", "precast" --> "PRECAST", "STAligner" --> "STAligner", "paste" --> "PASTE", "GraphST" --> "GraphST"
for method in ["prime", "precast", "STAligner", "paste", "GraphST"]:
    results_df.rename(index={f"X_{method}": method.upper() if method in ['prime', 'precast', 'paste'] else method}, inplace=True)    
# %%
from prime.plotting import plot_scib_results_table, save_scib_results_publication_pdf

fig, ax, table = plot_scib_results_table(
    results_df,
    show=False,
)
fig
# %%
save_scib_results_publication_pdf(
    fig, ax,
    save_path=f"{results_dir}/benchmark_results_table.pdf"
)
# %%

