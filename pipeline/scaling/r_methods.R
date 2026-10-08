# R integration methods timed by run_scaling.py (same subsets and meter as the Python methods).
#   Rscript r_methods.R <subset.h5ad> <out.npy> <stages.json> fastmnn|rpca
# Settings follow benchmark/r_methods/r_batch_correction_pipeline.R: 2000 HVGs, 50 dimensions, FastMNN k = 20,
# RPCA k.weight = 20. preprocess_s covers reading the h5ad and the method's own preprocessing.
suppressPackageStartupMessages({
  library(hdf5r); library(Matrix); library(Seurat); library(SeuratWrappers); library(RcppCNPy); library(jsonlite)
})
# Seurat v5 IntegrateLayers passes multi-GB globals to future even under the sequential plan;
# the default 500 MiB cap makes every RPCA run abort before integrating. No effect on results.
options(future.globals.maxSize = Inf)
a <- commandArgs(trailingOnly = TRUE)
method <- a[4]
t0 <- Sys.time()

h <- H5File$new(a[1], mode = "r")
read_col <- function(node) {                 # categorical group or plain string array
  if (inherits(node, "H5Group")) node[["categories"]]$read()[node[["codes"]]$read() + 1L] else node$read()
}
X <- h[["X"]]
shape <- h5attr(X, "shape")                   # cells x genes, CSR  ==  genes x cells, CSC
counts <- sparseMatrix(i = X[["indices"]]$read() + 1L, p = X[["indptr"]]$read(),
                       x = as.numeric(X[["data"]]$read()), dims = c(shape[2], shape[1]))
cells <- h[["obs"]][["_index"]]$read()
dimnames(counts) <- list(make.unique(h[["var"]][["_index"]]$read()), cells)
batch <- read_col(h[["obs"]][["batch"]])
h$close_all()

obj <- CreateSeuratObject(counts, meta.data = data.frame(batch = batch, row.names = cells))
rm(counts); invisible(gc())
if (method == "fastmnn") {
  obj <- NormalizeData(obj, verbose = FALSE)
  obj <- FindVariableFeatures(obj, nfeatures = 2000, verbose = FALSE)
  t1 <- Sys.time()
  obj <- RunFastMNN(object.list = SplitObject(obj, split.by = "batch"), features = VariableFeatures(obj),
                    reduction.name = "mnn", d = 50, k = 20, verbose = FALSE)
  emb <- Embeddings(obj, "mnn")
} else if (method == "rpca") {
  obj[["RNA"]] <- split(obj[["RNA"]], f = obj$batch)
  obj <- NormalizeData(obj, verbose = FALSE)
  obj <- FindVariableFeatures(obj, nfeatures = 2000, verbose = FALSE)
  obj <- ScaleData(obj, features = VariableFeatures(obj), verbose = FALSE)
  obj <- RunPCA(obj, features = VariableFeatures(obj), npcs = 50, verbose = FALSE)
  t1 <- Sys.time()
  obj <- IntegrateLayers(obj, method = RPCAIntegration, orig.reduction = "pca", new.reduction = "integrated.rpca",
                         dims = 1:50, k.weight = 20, verbose = FALSE)
  emb <- Embeddings(obj, "integrated.rpca")
} else stop("unknown method ", method)
t2 <- Sys.time()

emb <- emb[cells, , drop = FALSE]
storage.mode(emb) <- "double"
npySave(a[2], emb)
write_json(list(shared_preprocess_s = as.numeric(difftime(t1, t0, units = "secs")),
                core_s = as.numeric(difftime(t2, t1, units = "secs"))), a[3], auto_unbox = TRUE)
