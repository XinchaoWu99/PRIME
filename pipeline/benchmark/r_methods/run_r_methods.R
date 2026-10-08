#!/usr/bin/env Rscript
# Seurat-based integration methods (CCA, RPCA, joint PCA via IntegrateLayers; FastMNN via SeuratWrappers) on one h5ad.
#
#   Rscript run_r_methods.R <data.h5ad> <output_dir> <output_prefix> [batch_col=batch] [counts_path=layers/counts]
#                           [methods=cca,rpca,fastmnn,jpca] [workers=4] [max_globals_gb=100]
#
# The h5ad must hold raw counts at <counts_path> (an HDF5 path inside the file) and the batch labels in
# obs/<batch_col>. Every method uses 2,000 variable features, 50 dimensions, k = 20 (FastMNN) and k.weight = 20.
# Writes <output_dir>/<output_prefix>_<method>_corrected.npy (cells x 50, cell order of the h5ad) and
# <output_prefix>.rds (the Seurat object before integration).
#
# Calls used for the benchmarks (paths as in pipeline/config.yaml):
#   human immune          Rscript run_r_methods.R <immune dir>/Immune_ALL_human.h5ad <immune dir> Immune_ALL_human
#   lung adenocarcinoma   Rscript run_r_methods.R <lung dir>/benchmarked_lung_cancer_integrated.h5ad <lung dir> \
#                                 lung_cancer batch layers/count

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
source(file.path(script_dir, "r_batch_correction_pipeline.R"))

library(future)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("usage: Rscript run_r_methods.R <data.h5ad> <output_dir> <output_prefix> [batch_col] [counts_path] [methods] [workers] [max_globals_gb]")
}
arg <- function(i, default) if (length(args) >= i) args[i] else default
h5ad_file     <- args[1]
output_dir    <- args[2]
output_prefix <- args[3]
batch_col     <- arg(4, "batch")
counts_path   <- arg(5, "layers/counts")
methods       <- strsplit(arg(6, "cca,rpca,fastmnn,jpca"), ",")[[1]]
workers       <- as.integer(arg(7, "4"))
max_globals   <- as.numeric(arg(8, "100"))

options(future.globals.maxSize = max_globals * 1024^3)
plan(multisession, workers = workers)

cat(strrep("=", 60), "\n", sep = "")
cat("  File    :", h5ad_file, "\n")
cat("  Batch   :", batch_col, "\n")
cat("  Methods :", paste(methods, collapse = ", "), "\n")
cat(strrep("=", 60), "\n", sep = "")

pipeline_results <- run_h5ad_batch_correction_pipeline(
  h5ad_file     = h5ad_file,
  output_dir    = output_dir,
  output_prefix = output_prefix,
  batch_col     = batch_col,
  counts_path   = counts_path,
  assay         = "RNA",
  methods       = methods,
  nfeatures     = 2000,
  npcs          = 50,
  k_mnn         = 20,
  k_weight      = 20,
  save_rds      = TRUE,
  verbose       = FALSE
)

print(pipeline_results$outputs)
cat("\nRDS file:", pipeline_results$rds_file, "\n")
cat("Output files:\n")
cat(paste(pipeline_results$outputs$output_file, collapse = "\n"), "\n")
