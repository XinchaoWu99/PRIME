# One Monocle3 pipeline for every integration method (human immune data).
#
#   Rscript trajectory_comparison.R <all_embeddings.h5ad> <out_dir> [seeds=1,2,3,4,5] [methods] [root=single|all] [use_partition=TRUE|FALSE]
#
# <all_embeddings.h5ad> = the benchmark object of benchmark/immune/integrate_and_benchmark.py (obs: cell_type, batch;
# one obsm entry per method). `methods` is a comma-separated list of obsm keys (or `default`). The Python used by
# reticulate (anndata, scikit-learn) is taken from RETICULATE_PYTHON, or from the PATH.
#
# The comparison is run twice: <out_dir> = <config out_dir>/trajectory/partitioned with use_partition = TRUE and
# <config out_dir>/trajectory/one_graph with use_partition = FALSE (methods PRIME,Harmony,FastMNN, root single).
#
# Every method gets the same cells, the same PCA(50) view of its embedding, the
# same UMAP / clustering / learn_graph settings (package defaults, fixed below).
# Root:
#   single  one principal-graph node = the node most HSPC cells project to (the
#           procedure in the Monocle3 documentation). If a method scatters the HSPCs,
#           lineages outside the HSPC partition stay unreachable (pseudotime Inf).
#   all     every node closest to any HSPC cell. Scattered HSPCs
#           give every partition its own root, so reachability and "HSPC > X"
#           relations are satisfied trivially -- kept only as a diagnostic.
# use_partition: Monocle3 default TRUE (one principal graph per UMAP partition; lineages in
#   other partitions are unreachable). FALSE learns one graph over all cells (sensitivity only).
# Reference lineages are used only for scoring, never passed to the inference. Outputs:
#   pseudotime_<method>_s<seed>.tsv.gz, metrics.tsv, marker_dynamics.tsv, stability.tsv,
#   hspc_embedding.tsv (HSPC cohesion in each embedding), umap_<method>_s<seed>.tsv.gz and
#   graph_<method>_s<seed>.tsv (UMAP, partition, principal graph; first seed only)

suppressPackageStartupMessages({
  library(monocle3); library(reticulate); library(igraph); library(Matrix)
})
if (!nzchar(Sys.getenv("RETICULATE_PYTHON"))) use_python(Sys.which("python"), required = TRUE)
args <- commandArgs(trailingOnly = TRUE)
h5ad <- args[1]; out_dir <- args[2]
seeds <- if (length(args) > 2) as.integer(strsplit(args[3], ",")[[1]]) else 1:5
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

methods <- c("Unintegrated", "PRIME", "Harmony", "FastMNN", "scVI")
if (length(args) > 3 && args[4] != "default") methods <- strsplit(args[4], ",")[[1]]
root_mode <- if (length(args) > 4) args[5] else "single"
stopifnot(root_mode %in% c("single", "all"))
use_part <- if (length(args) > 5) as.logical(args[6]) else TRUE
root_type <- "HSPCs"
# The B-cell branch (CD10+ B -> CD20+ B -> plasma) is not scored as an HSPC lineage; B cells,
# like T/NK/DC cells, count as off-lineage when reached from the root.
lineages <- list(   # ordered stages; coarse reference, used for scoring only
  erythroid = c("HSPCs", "Erythroid progenitors", "Erythrocytes"),
  megakaryocyte = c("HSPCs", "Megakaryocyte progenitors"),
  monocyte = c("HSPCs", "Monocyte progenitors", "CD14+ Monocytes", "CD16+ Monocytes"))
markers <- list(    # expected sign of correlation with pseudotime inside the lineage
  erythroid = c(CD34 = -1, GATA1 = 1, KLF1 = 1, HBB = 1, HBA1 = 1),
  megakaryocyte = c(CD34 = -1, PF4 = 1, GP9 = 1),
  monocyte = c(CD34 = -1, MPO = 1, LYZ = 1, CD14 = 1, FCGR3A = 1))
genes <- unique(unlist(lapply(markers, names)))

py_run_string("   # r.<name> reads R variables

import anndata as ad, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
a = ad.read_h5ad(r.h5ad, backed='r')
obs = a.obs[['cell_type', 'batch']].astype(str)
genes = [g for g in list(r.genes) if g in a.var_names]
expr = a[:, genes].X
expr = expr.toarray() if hasattr(expr, 'toarray') else np.asarray(expr)
def view(Z):
    Z = np.asarray(Z.toarray() if hasattr(Z, 'toarray') else Z, dtype=np.float64)
    return PCA(50, random_state=0).fit_transform(Z) if Z.shape[1] > 50 else Z
embs = {m: view(a.obsm[m]) for m in list(r.methods) if m in a.obsm}

# HSPC cohesion in each embedding (before any UMAP): of the 15 nearest neighbours of an
# HSPC, how many are HSPCs, and how many of those HSPC neighbours come from another batch
ct = obs['cell_type'].values; bt = obs['batch'].values; h = np.where(ct == r.root_type)[0]
rows = []
for m, Z in embs.items():
    nn = NearestNeighbors(n_neighbors=16).fit(Z).kneighbors(Z[h], return_distance=False)[:, 1:]
    same = ct[nn] == r.root_type
    cross = same & (bt[nn] != bt[h][:, None])
    rows.append(dict(method=m, n_hspc=len(h), n_hspc_batches=len(set(bt[h])),
                     hspc_nn_same_type=same.mean(),
                     hspc_nn_cross_batch_given_hspc=cross.sum() / max(same.sum(), 1)))
hspc_emb = pd.DataFrame(rows)
")
obs <- py$obs; expr <- py$expr; colnames(expr) <- py$genes; rownames(expr) <- rownames(obs)
cells <- rownames(obs); ctype <- obs$cell_type
write.table(py$hspc_emb, file.path(out_dir, "hspc_embedding.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

ref_pairs <- unique(unlist(lapply(lineages, function(l)
  unlist(lapply(seq_along(l)[-length(l)], function(i) paste(l[i], l[(i + 1):length(l)], sep = ">"))))))
scored_types <- unique(unlist(lineages))

ancestor_pairs <- function(cds) {
  aux <- cds@principal_graph_aux[["UMAP"]]
  g <- principal_graph(cds)[["UMAP"]]
  v <- paste0("Y_", aux$pr_graph_cell_proj_closest_vertex[cells, 1])
  rep_v <- sapply(scored_types, function(t) names(which.max(table(v[ctype == t]))))
  d <- distances(g, v = aux$root_pr_nodes, to = V(g))
  out <- character(0)
  for (b in setdiff(scored_types, root_type)) {
    vb <- rep_v[[b]]
    if (!any(is.finite(d[, vb]))) next
    out <- c(out, paste(root_type, b, sep = ">"))
    r <- aux$root_pr_nodes[which.min(d[, vb])]
    path <- names(shortest_paths(g, from = r, to = vb)$vpath[[1]])
    for (a in setdiff(scored_types, c(b, root_type)))
      if (rep_v[[a]] %in% path && rep_v[[a]] != vb) out <- c(out, paste(a, b, sep = ">"))
  }
  unique(out)
}

metrics <- list(); dyn <- list(); pts <- list()
cds0 <- new_cell_data_set(as(t(expr), "dgCMatrix"), cell_metadata = obs,
                          gene_metadata = data.frame(gene_short_name = colnames(expr), row.names = colnames(expr)))
is_root <- ctype == root_type
for (m in names(py$embs)) for (s in seeds) {
  cds <- cds0
  Z <- py$embs[[m]]; rownames(Z) <- cells
  SingleCellExperiment::reducedDims(cds)[["PCA"]] <- Z
  set.seed(s)
  cds <- reduce_dimension(cds, reduction_method = "UMAP", preprocess_method = "PCA",
                          umap.fast_sgd = FALSE, cores = 1, verbose = FALSE)
  cds <- cluster_cells(cds, reduction_method = "UMAP", random_seed = s)
  cds <- learn_graph(cds, use_partition = use_part, close_loop = FALSE, verbose = FALSE)
  closest <- cds@principal_graph_aux[["UMAP"]]$pr_graph_cell_proj_closest_vertex[cells, 1]
  vnames <- igraph::V(principal_graph(cds)[["UMAP"]])$name
  if (root_mode == "single") {
    cds <- order_cells(cds, root_pr_nodes = vnames[as.integer(names(which.max(table(closest[is_root]))))])
  } else {
    cds <- order_cells(cds, root_cells = cells[is_root])
  }
  pt <- pseudotime(cds)[cells]
  pts[[paste(m, s)]] <- pt
  write.table(data.frame(cell = cells, pseudotime = pt), gzfile(file.path(out_dir, sprintf("pseudotime_%s_s%d.tsv.gz", m, s))),
              sep = "\t", quote = FALSE, row.names = FALSE)

  part <- as.character(partitions(cds)[cells])
  hp <- sort(table(part[is_root]), decreasing = TRUE)
  row <- data.frame(method = m, seed = s, root_mode = root_mode, use_partition = use_part, coverage = mean(is.finite(pt)),
                    n_partitions = length(unique(part)),
                    hspc_partitions = length(hp), hspc_frac_top_partition = hp[[1]] / sum(is_root),
                    n_root_nodes_all_hspc = length(unique(closest[is_root])),
                    hspc_reachable = mean(is.finite(pt[is_root])),
                    off_lineage_reached = mean(is.finite(pt[!(ctype %in% unlist(lineages))])))
  pred <- ancestor_pairs(cds); tp <- length(intersect(pred, ref_pairs))
  row$topo_precision <- tp / max(length(pred), 1); row$topo_recall <- tp / length(ref_pairs)
  row$topo_f1 <- with(row, ifelse(topo_precision + topo_recall > 0,
                                  2 * topo_precision * topo_recall / (topo_precision + topo_recall), 0))
  for (ln in names(lineages)) {
    k <- ctype %in% lineages[[ln]] & is.finite(pt)
    stage <- match(ctype[k], lineages[[ln]])
    row[[paste0("spearman_", ln)]] <- suppressWarnings(cor(pt[k], stage, method = "spearman"))
    row[[paste0("coverage_", ln)]] <- mean(is.finite(pt[ctype %in% lineages[[ln]]]))
    for (g in intersect(names(markers[[ln]]), colnames(expr))) {
      r <- suppressWarnings(cor(expr[k, g], pt[k], method = "spearman"))
      dyn[[length(dyn) + 1]] <- data.frame(method = m, seed = s, lineage = ln, gene = g, rho = r,
                                           expected_sign = markers[[ln]][[g]], sign_ok = sign(r) == markers[[ln]][[g]])
    }
  }
  metrics[[length(metrics) + 1]] <- row

  if (s == seeds[1]) {   # coordinates for plotting / visual checks
    U <- SingleCellExperiment::reducedDims(cds)[["UMAP"]][cells, ]
    write.table(data.frame(cell = cells, umap1 = U[, 1], umap2 = U[, 2], cell_type = ctype, batch = obs$batch,
                           partition = part, pseudotime = pt),
                gzfile(file.path(out_dir, sprintf("umap_%s_s%d.tsv.gz", m, s))), sep = "\t", quote = FALSE, row.names = FALSE)
    g <- principal_graph(cds)[["UMAP"]]; Y <- t(cds@principal_graph_aux[["UMAP"]]$dp_mst)
    ed <- igraph::as_edgelist(g)
    write.table(data.frame(x1 = Y[ed[, 1], 1], y1 = Y[ed[, 1], 2], x2 = Y[ed[, 2], 1], y2 = Y[ed[, 2], 2],
                           root = ed[, 1] %in% cds@principal_graph_aux[["UMAP"]]$root_pr_nodes |
                             ed[, 2] %in% cds@principal_graph_aux[["UMAP"]]$root_pr_nodes),
                file.path(out_dir, sprintf("graph_%s_s%d.tsv", m, s)), sep = "\t", quote = FALSE, row.names = FALSE)
  }
}

stab <- if (length(seeds) < 2) data.frame(method = names(py$embs), mean_seed_spearman = NA) else
  do.call(rbind, lapply(names(py$embs), function(m) {
  prs <- combn(seeds, 2)
  data.frame(method = m, mean_seed_spearman = mean(apply(prs, 2, function(p) {
    a <- pts[[paste(m, p[1])]]; b <- pts[[paste(m, p[2])]]; k <- is.finite(a) & is.finite(b)
    cor(a[k], b[k], method = "spearman")
  })))
}))
write.table(do.call(rbind, metrics), file.path(out_dir, "metrics.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(do.call(rbind, dyn), file.path(out_dir, "marker_dynamics.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(stab, file.path(out_dir, "stability.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
