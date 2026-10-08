# PRECAST on the 12 DLPFC sections (K = 7, 2,000 HVGs, default embedding dimension 15).
#
#   Rscript run_precast.R <dlpfc_data_dir> <results_dir> [K=7]
#
# <dlpfc_data_dir> holds one Space Ranger folder per section (filtered_feature_bc_matrix.h5, metadata.tsv with the
# columns barcode, row, col, layer_guess) = config.yaml `benchmark.dlpfc.data_dir`; <results_dir> =
# `benchmark.dlpfc.results_dir`. Official workflow: CreatePRECASTObject -> AddAdjList(Visium) -> AddParSetting ->
# PRECAST -> SelectModel -> IntegrateSpaData. Writes PRECAST_embedding.npy and PRECAST_embedding_with_meta.csv
# (embedding, section, barcode). reticulate needs a Python with numpy.

suppressPackageStartupMessages({
  library(Seurat)
  library(PRECAST)
  library(reticulate)
})

# ---- args ----
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript run_precast.R <dlpfc_data_dir> <results_dir> [K=7]")
data_root <- args[1]
out_dir   <- args[2]
K         <- if (length(args) > 2) as.integer(args[3]) else 7L

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

section_ids <- c("151507", "151508", "151509", "151510",
                 "151669", "151670", "151671", "151672",
                 "151673", "151674", "151675", "151676")


# =====================================================================
# 1. Load each section -> Seurat object with `row`, `col` in meta.data
# =====================================================================
read_section <- function(sec_dir, sec) {
  h5_path   <- file.path(sec_dir, "filtered_feature_bc_matrix.h5")
  meta_path <- file.path(sec_dir, "metadata.tsv")

  counts <- Read10X_h5(h5_path)
  # In rare multi-modality h5 files Read10X_h5 returns a list; pick GEX
  if (is.list(counts) && !inherits(counts, "dgCMatrix")) {
    counts <- counts[["Gene Expression"]]
  }

  meta <- read.table(meta_path, header = TRUE, sep = "\t",
                     stringsAsFactors = FALSE, check.names = FALSE,
                     comment.char = "")
  # Set barcode as rownames (DLPFC metadata.tsv usually has a `barcode` column)
  bc_col <- intersect(c("barcode", "Barcode", "barcodes"), colnames(meta))
  if (length(bc_col) >= 1) {
    rownames(meta) <- meta[[bc_col[1]]]
  }
  stopifnot(all(c("row", "col") %in% colnames(meta)))   # PRECAST requirement

  # Keep only spots present in BOTH counts and metadata
  common <- intersect(colnames(counts), rownames(meta))
  if (length(common) == 0) {
    stop("No barcodes shared between counts and metadata for section ", sec)
  }
  counts <- counts[, common]
  meta   <- meta[common, , drop = FALSE]
  meta$section_id <- sec

  seu <- CreateSeuratObject(counts = counts, meta.data = meta,
                            project = sec, min.cells = 10, min.features = 10)
  seu
}

cat("Loading", length(section_ids), "sections from", data_root, "\n")
seuList <- vector("list", length(section_ids))
names(seuList) <- section_ids
for (sec in section_ids) {
  cat("  ", sec, "...")
  seuList[[sec]] <- read_section(file.path(data_root, sec), sec)
  cat("  ", ncol(seuList[[sec]]), "spots\n")
}

# =====================================================================
# 2. PRECAST workflow
#    (CreatePRECASTObject -> AddAdjList -> AddParSetting -> PRECAST ->
#     SelectModel -> IntegrateSpaData)
# =====================================================================
set.seed(2023)
cat("\nCreating PRECAST object (HVGs = 2000)...\n")
preobj <- CreatePRECASTObject(seuList = seuList,
                              selectGenesMethod = "HVGs",

                              gene.number = 2000)

cat("Adding spatial adjacency (Visium platform)...\n")
PRECASTObj <- AddAdjList(preobj, platform = "Visium")

cat("Adding model setting (mclust init, maxIter = 30, coreNum = 4)...\n")
PRECASTObj <- AddParSetting(PRECASTObj,
                            Sigma_equal = TRUE,
                            coreNum     = 4,
                            int.model   = "mclust",
                            maxIter     = 30,
                            verbose     = TRUE)

cat("Fitting PRECAST with K =", K, "...\n")
PRECASTObj <- PRECAST(PRECASTObj, K = K)
PRECASTObj <- SelectModel(PRECASTObj)

cat("Integrating spatial data (Human housekeeping genes)...\n")
seuInt <- IntegrateSpaData(PRECASTObj, species = "Human")
cat("Integration done. seuInt:", ncol(seuInt), "spots,",
    length(unique(seuInt$batch)), "batches\n")

meta_master <- do.call(rbind, lapply(names(seuList), function(sec) {
  m <- seuList[[sec]]@meta.data
  m$section_id        <- sec
  m$original_barcode  <- rownames(m)
  m
}))

batch_idx        <- as.integer(as.character(seuInt$batch))
section_per_spot <- names(seuList)[batch_idx]
barcodes_clean   <- sub("-\\d+$", "", colnames(seuInt))   # strip _1/_2 suffix

key_master <- paste(meta_master$section_id, sub("-\\d+$", "", meta_master$original_barcode), sep = "|")
key_int    <- paste(section_per_spot,       barcodes_clean,               sep = "|")
ord        <- match(key_int, key_master)

if (any(is.na(ord))) {
  warning(sum(is.na(ord)), " spots in seuInt could not be matched to seuList ",
          "metadata; their re-merged columns will be NA.")
}

meta_aligned <- meta_master[ord, , drop = FALSE]
rownames(meta_aligned) <- colnames(seuInt)

cols_to_add <- setdiff(colnames(meta_aligned), colnames(seuInt@meta.data))

# Check rows align first
stopifnot(all(rownames(seuInt@meta.data) %in% rownames(meta_aligned)))

for (col in cols_to_add) {
  seuInt@meta.data[[col]] <- meta_aligned[rownames(seuInt@meta.data), col]
}

# Verify
cat("New columns added:", length(cols_to_add), "\n")
cat("Total meta.data columns:", ncol(seuInt@meta.data), "\n")

# =====================================================================
# 3. Extract embedding + clusters
# =====================================================================
embedding <- Embeddings(seuInt, reduction = "PRECAST")   # (n, q)
clusters  <- as.integer(as.character(seuInt$cluster))    # 1..K

# Map seuInt's `batch` (1..n_sections) back to section_id strings.
# CreatePRECASTObject preserves seuList order, so batch i -> section_ids[i].
batch_idx <- as.integer(as.character(seuInt$batch))
section_per_spot <- section_ids[batch_idx]

# seuInt may have appended _1 / _2 suffixes to barcodes for uniqueness.
# Strip them to recover the original Visium barcodes.
barcodes_clean <- sub("_\\d+$", "", colnames(seuInt))

output_meta <- data.frame(
  section_id  = section_per_spot,
  barcode     = barcodes_clean,
  layer_guess = seuInt@meta.data[["layer_guess"]],
  cluster     = clusters,
  stringsAsFactors = FALSE
)

cat("Embedding dim:", nrow(embedding), "x", ncol(embedding), "\n")

embedding_df <- as.data.frame(embedding)
embedding_df$section <- seuInt@meta.data[rownames(embedding_df), "section_id"]   # adjust column name
embedding_df$barcode  <- seuInt@meta.data[rownames(embedding_df), "barcode"]  # adjust column name

# =====================================================================
# 4. Save via reticulate (numpy)
# =====================================================================
np <- import("numpy", convert = FALSE)

np$save(
  file.path(out_dir, "PRECAST_embedding.npy"),
  np$asarray(r_to_py(embedding), dtype = "float32")
)

write.csv(embedding_df, file.path(out_dir, "PRECAST_embedding_with_meta.csv"), row.names = FALSE)
