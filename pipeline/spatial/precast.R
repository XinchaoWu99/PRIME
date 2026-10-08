# PRECAST on a set of sections (called by run_methods.py).
#   Rscript precast.R <DLPFC dir | data.h5ad> <out.csv> <seed> <K> <n_hvg> [q=15]
# Input: the folder holding one Space Ranger folder per DLPFC section, or an h5ad (Visium HD bins).
# Official workflow: CreatePRECASTObject -> AddAdjList(Visium) -> AddParSetting ->
# PRECAST -> SelectModel -> IntegrateSpaData. Output: section, barcode, PRECAST_1..q.
suppressPackageStartupMessages({ library(Seurat); library(PRECAST) })
a <- commandArgs(trailingOnly = TRUE)
root <- a[1]; out <- a[2]; seed <- as.integer(a[3]); K <- as.integer(a[4]); n_hvg <- as.integer(a[5])
q <- if (length(a) > 5) as.integer(a[6]) else 15L
if (grepl("\\.h5ad$", root)) {      # h5ad input (Visium HD): counts + obs batch / array_row / array_col
  library(hdf5r)
  h <- H5File$new(root, mode = "r")
  col <- function(n) { x <- h[["obs"]][[n]]; if (inherits(x, "H5Group")) x[["categories"]]$read()[x[["codes"]]$read() + 1L] else x$read() }
  X <- h[["X"]]; shape <- h5attr(X, "shape")
  counts <- Matrix::sparseMatrix(i = X[["indices"]]$read() + 1L, p = X[["indptr"]]$read(),
                                 x = as.numeric(X[["data"]]$read()), dims = c(shape[2], shape[1]))
  cells <- h[["obs"]][["_index"]]$read()
  dimnames(counts) <- list(make.unique(h[["var"]][["_index"]]$read()), cells)
  meta <- data.frame(batch = col("batch"), row = col("array_row"), col = col("array_col"), row.names = cells)
  h$close_all()
  sections <- sort(unique(meta$batch))
  seuList <- lapply(sections, function(s) {
    k <- meta$batch == s
    CreateSeuratObject(counts[, k], meta.data = meta[k, ])
  })
  adj_type <- "fixed_number"            # square bin grid: 6 nearest bins, like the 6 Visium neighbours
} else {
  sections <- sort(list.dirs(root, full.names = FALSE, recursive = FALSE))
  seuList <- lapply(sections, function(s) {
    counts <- Read10X_h5(file.path(root, s, "filtered_feature_bc_matrix.h5"))
    if (is.list(counts)) counts <- counts[["Gene Expression"]]
    meta <- read.delim(file.path(root, s, "metadata.tsv"), check.names = FALSE)
    rownames(meta) <- meta$barcode
    meta <- meta[intersect(colnames(counts), rownames(meta)), ]
    obj <- CreateSeuratObject(counts[, rownames(meta)], meta.data = meta)
    obj$section <- s
    obj
  })
  adj_type <- "fixed_distance"
}
names(seuList) <- sections

set.seed(seed)
obj <- CreatePRECASTObject(seuList, selectGenesMethod = "HVGs", gene.number = n_hvg)
obj <- if (adj_type == "fixed_number") AddAdjList(obj, type = "fixed_number", number = 6) else AddAdjList(obj, platform = "Visium")
obj <- AddParSetting(obj, Sigma_equal = TRUE, coreNum = 4, int.model = "mclust", maxIter = 30, seed = seed, verbose = FALSE)
obj <- PRECAST(obj, K = K, q = q)
obj <- SelectModel(obj)
seuInt <- IntegrateSpaData(obj, species = "Human")

emb <- Embeddings(seuInt, reduction = "PRECAST")
section <- sections[as.integer(as.character(seuInt$batch))]
barcode <- sub("_\\d+$", "", colnames(seuInt))
write.csv(data.frame(section = section, barcode = barcode, emb, check.names = FALSE), out, row.names = FALSE)
