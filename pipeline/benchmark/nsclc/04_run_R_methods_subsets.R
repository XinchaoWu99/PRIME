# Robustness experiment, step 2/3 — R-based integration per subset.
#
# Same pipeline as ../r_methods/run_r_methods.R, but instead of one data set it loops over
# every subset.h5ad produced by 03_make_dataset_subsets.py and writes the
# corrected embeddings (CCA / FastMNN / jPCA / RPCA) into each subset folder as
# subset_<method>_corrected.npy.
#
# Run order:  03 (Python)  ->  04 (this)  ->  05 (Python integration)  ->  06 (benchmark).
#
# Usage:
#   Rscript 04_run_R_methods_subsets.R <robustness_dir>
# <robustness_dir> = <benchmark.lung.dir of config.yaml>/robustness (written by 03).

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
source(file.path(script_dir, "..", "r_methods", "r_batch_correction_pipeline.R"))

library(future)

options(future.globals.maxSize = 300 * 1024^3)  # 300 GiB
plan(multisession, workers = 4)

# ---------------------------------------------------------------------------
# Settings (keep in sync with 03 / 05)
# ---------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript 04_run_R_methods_subsets.R <robustness_dir>")
robust_root  <- args[1]
manifest_csv <- file.path(robust_root, "subset_manifest.csv")
output_prefix <- "subset"          # -> subset_<method>_corrected.npy

pipeline_config <- list(
  counts_path = "layers/count",
  assay       = "RNA",
  methods     = c("fastmnn", "cca", "rpca", "jpca"),
  nfeatures   = 2000,
  npcs        = 50,
  k_mnn       = 20,
  k_weight    = 20,
  verbose     = FALSE
)

# ---------------------------------------------------------------------------
# Discover subsets (prefer the manifest; fall back to globbing)
# ---------------------------------------------------------------------------

if (file.exists(manifest_csv)) {
  manifest <- read.csv(manifest_csv, stringsAsFactors = FALSE)
  subset_dirs <- file.path(robust_root, manifest$subset_id)
} else {
  subset_dirs <- list.dirs(robust_root, recursive = FALSE)
  subset_dirs <- subset_dirs[file.exists(file.path(subset_dirs, "subset.h5ad"))]
}

cat("Found", length(subset_dirs), "subset(s) under", robust_root, "\n")

# ---------------------------------------------------------------------------
# Run the R pipeline for every subset
# ---------------------------------------------------------------------------

for (sub_dir in subset_dirs) {
  subset_id <- basename(sub_dir)
  h5ad_file <- file.path(sub_dir, "subset.h5ad")

  if (!file.exists(h5ad_file)) {
    cat("[skip]", subset_id, "- subset.h5ad not found\n")
    next
  }

  # Resume support: skip if all expected .npy outputs already exist.
  expected <- file.path(
    sub_dir, sprintf("%s_%s_corrected.npy", output_prefix, pipeline_config$methods)
  )
  if (all(file.exists(expected))) {
    cat("[exists]", subset_id, "- all R embeddings present, skipping\n")
    next
  }

  cat("\n", strrep("=", 60), "\n", sep = "")
  cat("  Subset :", subset_id, "\n")
  cat("  File   :", h5ad_file, "\n")
  cat(strrep("=", 60), "\n", sep = "")

  pipeline_results <- do.call(
    run_h5ad_batch_correction_pipeline,
    c(
      list(
        h5ad_file     = h5ad_file,
        output_dir    = sub_dir,
        output_prefix = output_prefix,
        batch_col     = "batch",
        save_rds      = FALSE
      ),
      pipeline_config
    )
  )

  cat("\n=== Results for", subset_id, "===\n")
  print(pipeline_results$outputs)
}

cat("\nAll subsets completed.\n")
