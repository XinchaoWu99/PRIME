#!/usr/bin/env Rscript
# Monocle3 trajectories of the human immune data, one run per integration method, drawn on the method's own UMAP.
# Settings: cluster_cells k = 15 on the method's UMAP; learn_graph ncenter = 150, nn.k = 30, minimal_branch_len = 20,
# use_partition = TRUE; root = all HSPCs; fixed seed.
#
#   Rscript trajectory_panels.R <all_embeddings.h5ad> <umap_dir> <out_dir> <method> [<method> ...]
#
# <all_embeddings.h5ad> = the benchmark object of benchmark/immune/integrate_and_benchmark.py.
# <umap_dir>/Immune_ALL_human_<method in lower case>_umap.npy: 2-D UMAPs of the method's embedding
# (benchmark/immune/integrate_and_benchmark.py, or recomputed by notebooks/01_human_immune.ipynb).
# Writes <out_dir>/<method>/pseudotime_embedding.pdf, cell_type_embedding.pdf and cds_colData_with_pseudotime.csv.
suppressPackageStartupMessages({
  library(reticulate)
  library(ggplot2)
})

library(hdf5r)
library(Matrix)
library(Seurat)
library(monocle3)

to_chr <- function(x) {
  if (is.null(x)) return(character())
  if (is.factor(x)) return(as.character(x))
  if (is.list(x)) return(vapply(x, as.character, character(1)))
  as.character(x)
}

read_h5ad_vector <- function(node) {
  enc <- if ("encoding-type" %in% hdf5r::h5attr_names(node)) {
    hdf5r::h5attr(node, "encoding-type")
  } else {
    NULL
  }

  if (inherits(node, "H5D")) {
    values <- node$read()
    if (identical(enc, "string-array")) values <- to_chr(values)
    return(values)
  }

  if (!inherits(node, "H5Group")) stop("Unsupported node type")

  if (identical(enc, "categorical")) {
    categories <- to_chr(node[["categories"]]$read())
    codes <- as.integer(node[["codes"]]$read())
    values <- rep(NA_character_, length(codes))
    keep <- codes >= 0L
    values[keep] <- categories[codes[keep] + 1L]
    ordered <- if ("ordered" %in% hdf5r::h5attr_names(node)) {
      isTRUE(hdf5r::h5attr(node, "ordered"))
    } else {
      FALSE
    }
    return(factor(values, levels = categories, ordered = ordered))
  }

  if (identical(enc, "nullable-string-array")) {
    values <- to_chr(node[["values"]]$read())
    mask <- as.logical(node[["mask"]]$read())
    values[mask] <- NA_character_
    return(values)
  }

  if (identical(enc, "nullable-boolean")) {
    values <- as.logical(node[["values"]]$read())
    mask <- as.logical(node[["mask"]]$read())
    values[mask] <- NA
    return(values)
  }

  if (identical(enc, "nullable-integer")) {
    values <- node[["values"]]$read()
    mask <- as.logical(node[["mask"]]$read())
    values[mask] <- NA
    return(values)
  }

  stop("Unsupported encoding-type: ", enc)
}

read_h5ad_dataframe <- function(group) {
  index_key <- as.character(hdf5r::h5attr(group, "_index"))
  column_order <- as.character(hdf5r::h5attr(group, "column-order"))
  index_values <- to_chr(read_h5ad_vector(group[[index_key]]))

  if (length(column_order) == 0) {
    return(data.frame(row.names = make.unique(index_values)))
  }

  columns <- vector("list", length(column_order))
  names(columns) <- column_order

  for (column_name in column_order) {
    columns[[column_name]] <- read_h5ad_vector(group[[column_name]])
  }

  data_frame <- as.data.frame(columns, stringsAsFactors = FALSE, optional = TRUE)
  rownames(data_frame) <- make.unique(index_values)
  data_frame
}

read_h5ad_matrix <- function(node) {
  enc <- if ("encoding-type" %in% hdf5r::h5attr_names(node)) {
    hdf5r::h5attr(node, "encoding-type")
  } else {
    NULL
  }

  if (inherits(node, "H5D")) {
    return(t(node$read()))
  }

  if (!inherits(node, "H5Group")) stop("Unsupported matrix node type")

  shape <- as.integer(hdf5r::h5attr(node, "shape"))
  values <- node[["data"]]$read()
  indices <- as.integer(node[["indices"]]$read())
  indptr <- as.integer(node[["indptr"]]$read())

  if (identical(enc, "csr_matrix")) {
    matrix_data <- sparseMatrix(
      j = indices,
      p = indptr,
      x = values,
      dims = shape,
      index1 = FALSE,
      repr = "R"
    )
    return(t(matrix_data))
  }

  if (identical(enc, "csc_matrix")) {
    matrix_data <- sparseMatrix(
      i = indices,
      p = indptr,
      x = values,
      dims = shape,
      index1 = FALSE,
      repr = "C"
    )
    return(t(matrix_data))
  }

  stop("Unsupported matrix encoding: ", enc)
}

make_seurat_from_h5ad <- function(file, counts_path = "layers/counts", assay = "RNA") {
  h5_file <- H5File$new(file, mode = "r")
  on.exit(h5_file$close_all(), add = TRUE)

  obs <- read_h5ad_dataframe(h5_file[["obs"]])
  var <- read_h5ad_dataframe(h5_file[["var"]])
  counts <- t(read_h5ad_matrix(h5_file[[counts_path]]))

  colnames(counts) <- make.unique(rownames(obs))
  rownames(counts) <- make.unique(rownames(var))
  rownames(obs) <- colnames(counts)

  object <- CreateSeuratObject(
    counts = counts,
    meta.data = obs,
    assay = assay
  )

  list(object = object, obs = obs, var = var)
}

seurat_to_monocle3 <- function(seurat_obj, assay = "RNA") {
  expr_mat <- GetAssayData(seurat_obj, assay = assay, layer = "counts")
  cell_metadata <- seurat_obj[[]]
  gene_metadata <- data.frame(
    gene_short_name = rownames(expr_mat),
    row.names = rownames(expr_mat),
    stringsAsFactors = FALSE
  )

  new_cell_data_set(
    expression_data = expr_mat,
    cell_metadata = cell_metadata,
    gene_metadata = gene_metadata
  )
}

args <- commandArgs(trailingOnly = TRUE)
h5ad_file <- args[1]; umap_dir <- args[2]; out_root <- args[3]; methods <- args[-(1:3)]
set.seed(2026)

seurat <- make_seurat_from_h5ad(h5ad_file, counts_path = "layers/counts")
cds <- seurat_to_monocle3(seurat$object)
cds <- preprocess_cds(cds, num_dim = 50)
np <- import("numpy")
for (method in methods) {
  umap_mat <- as.matrix(py_to_r(np$load(file.path(umap_dir, sprintf("Immune_ALL_human_%s_umap.npy", tolower(method))))))
  storage.mode(umap_mat) <- "double"
  rownames(umap_mat) <- colnames(cds)
  colnames(umap_mat) <- c("UMAP_1", "UMAP_2")
  reducedDims(cds)[[method]] <- umap_mat
}

for (method in methods) {
  message("trajectory: ", method)
  out_dir <- file.path(out_root, method)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  reducedDims(cds)$UMAP <- reducedDims(cds)[[method]]
  cds <- cluster_cells(cds, reduction_method = "UMAP", k = 15, nn_control = list(method = "nn2"))
  cds <- learn_graph(
    cds,
    use_partition = TRUE,
    learn_graph_control = list(ncenter = 150, nn.k = 30, minimal_branch_len = 20),
    close_loop = FALSE
  )
  root_cells <- colnames(cds)[colData(cds)$cell_type == "HSPCs"]
  cds <- order_cells(cds, root_cells = root_cells)

  p = plot_cells(
    cds,
    color_cells_by = "pseudotime",
    label_cell_groups=FALSE,
    label_leaves = FALSE,
    label_branch_points = FALSE,
    label_roots = FALSE,
    cell_size = 0.6,
    trajectory_graph_color = "green"
    )

  p2 = p +
    labs(
      title = sprintf("%s Pseudotime", method),
      x = "UMAP 1",
      y = "UMAP 2"
    ) +
    theme_classic(base_size = 18) +
    theme(
      text = element_text(size = 18, family = "Arial"),
      axis.title = element_text(size = 22),
      axis.text = element_text(size = 18),
      legend.title = element_text(size = 20),
      legend.text = element_text(size = 18),
      plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
      panel.border = element_blank(),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    )
  ggsave(file.path(out_dir, "pseudotime_embedding.pdf"), plot = p2, device = cairo_pdf, width = 10, height = 8)

  p = plot_cells(
    cds,
    color_cells_by = "cell_type",
    label_cell_groups=FALSE,
    label_leaves = FALSE,
    label_branch_points = FALSE,
    label_roots = FALSE,
    cell_size = 0.6,
    trajectory_graph_color = "green"
    )

  p2 = p +
    labs(
      title = sprintf("%s trajectory", method),
      x = "UMAP 1",
      y = "UMAP 2"
    ) +
    theme_classic(base_size = 18) +
    theme(
      text = element_text(size = 18, family = "Arial"),
      axis.title = element_text(size = 22),
      axis.text = element_text(size = 18),
      legend.title = element_text(size = 20),
      legend.text = element_text(size = 18),
      plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
      panel.border = element_blank(),
      plot.background = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA)
    )
  ggsave(file.path(out_dir, "cell_type_embedding.pdf"), plot = p2, device = cairo_pdf, width = 12, height = 8)

  cd <- as.data.frame(colData(cds))
  cd$pseudotime <- pseudotime(cds)
  write.csv(cd, file.path(out_dir, "cds_colData_with_pseudotime.csv"), row.names = TRUE)
}
