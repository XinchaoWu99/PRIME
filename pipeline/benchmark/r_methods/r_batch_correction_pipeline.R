library(hdf5r)
library(Matrix)
library(Seurat)
library(SeuratWrappers)
library(RcppCNPy)

to_chr <- function(x) {
  if (is.null(x)) {
    return(character())
  }
  if (is.factor(x)) {
    return(as.character(x))
  }
  if (is.list(x)) {
    return(vapply(x, as.character, character(1)))
  }
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
    if (identical(enc, "string-array")) {
      values <- to_chr(values)
    }
    return(values)
  }

  if (!inherits(node, "H5Group")) {
    stop("Unsupported node type")
  }

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

  if (!inherits(node, "H5Group")) {
    stop("Unsupported matrix node type")
  }

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
  # test which is obs and which is var

  counts <- read_h5ad_matrix(h5_file[[counts_path]])

  n_obs <- nrow(obs)
  n_var <- nrow(var)

  if (nrow(counts) == n_var && ncol(counts) == n_obs) {
    # Already in genes x cells orientation — nothing to do
  } else if (nrow(counts) == n_obs && ncol(counts) == n_var) {
    # Came out cells x genes — transpose to genes x cells
    counts <- t(counts)
  } else {
    stop(sprintf(
      paste0(
        "counts matrix dimensions (%d x %d) do not match ",
        "obs (%d) or var (%d) in any orientation.\n",
        "Check that counts_path = '%s' points to the correct layer."
      ),
      nrow(counts), ncol(counts), n_obs, n_var, counts_path
    ))
  }

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

validate_batch_column <- function(object, batch_col) {
  meta <- object[[]]
  if (!batch_col %in% colnames(meta)) {
    stop(
      sprintf(
        "Column '%s' was not found in obj[[]]. Available columns: %s",
        batch_col,
        paste(colnames(meta), collapse = ", ")
      )
    )
  }

  if (any(is.na(meta[[batch_col]]))) {
    stop(sprintf("Found NA values in obj[['%s']]. Please fix batch labels first.", batch_col))
  }

  invisible(TRUE)
}

prepare_seurat_for_integration <- function(
  object,
  batch_col,
  assay = "RNA",
  nfeatures = 2000,
  npcs = 50,
  verbose = FALSE
) {
  validate_batch_column(object, batch_col)
  object[[assay]] <- split(object[[assay]], f = object[[batch_col]][, 1])
  object <- NormalizeData(object, assay = assay, verbose = verbose)
  object <- FindVariableFeatures(
    object,
    assay = assay,
    nfeatures = nfeatures,
    verbose = verbose
  )
  features <- VariableFeatures(object, assay = assay)
  object <- ScaleData(object, assay = assay, features = features, verbose = verbose)
  object <- RunPCA(
    object,
    assay = assay,
    features = features,
    npcs = npcs,
    verbose = verbose
  )
  object
}

prepare_seurat_for_fastmnn <- function(
  object,
  assay = "RNA",
  nfeatures = 2000,
  verbose = FALSE
) {
  object <- NormalizeData(object, assay = assay, verbose = verbose)
  object <- FindVariableFeatures(
    object,
    assay = assay,
    nfeatures = nfeatures,
    verbose = verbose
  )
  object
}

prepare_seurat_panel_for_conos <- function(
  object,
  batch_col,
  assay = "RNA",
  nfeatures = 2000,
  npcs = 50,
  verbose = FALSE
) {
  validate_batch_column(object, batch_col)
  object.list <- SplitObject(object, split.by = batch_col)

  for (index in seq_along(object.list)) {
    sample_object <- object.list[[index]]
    sample_object <- NormalizeData(sample_object, assay = assay, verbose = verbose)
    sample_object <- FindVariableFeatures(
      sample_object,
      assay = assay,
      nfeatures = nfeatures,
      verbose = verbose
    )
    features <- VariableFeatures(sample_object, assay = assay)
    sample_object <- ScaleData(
      sample_object,
      assay = assay,
      features = features,
      verbose = verbose
    )
    sample_object <- RunPCA(
      sample_object,
      assay = assay,
      features = features,
      npcs = npcs,
      verbose = verbose
    )
    object.list[[index]] <- sample_object
  }

  object.list
}

save_embedding_npy <- function(embedding, file) {
  dir.create(dirname(file), recursive = TRUE, showWarnings = FALSE)
  storage.mode(embedding) <- "double"
  RcppCNPy::npySave(file, embedding)
  invisible(file)
}

run_with_future_strategy <- function(
  fun,
  future_strategy = c("sequential", "current"),
  future_globals_max_size = NULL
) {
  future_strategy <- match.arg(future_strategy)

  if (!requireNamespace("future", quietly = TRUE) || identical(future_strategy, "current")) {
    return(fun())
  }

  previous_plan <- future::plan()
  on.exit(future::plan(previous_plan), add = TRUE)

  if (!is.null(future_globals_max_size)) {
    previous_options <- options(future.globals.maxSize = future_globals_max_size)
    on.exit(options(previous_options), add = TRUE)
  }

  future::plan(future::sequential)
  fun()
}

available_r_batch_methods <- function() {
  methods <- list(
    fastmnn = list(
      reduction = "mnn",
      runner = function(object, batch_col, assay, nfeatures, npcs, k_mnn, k_weight, verbose) {
        prepared <- prepare_seurat_for_fastmnn(
          object = object,
          assay = assay,
          nfeatures = nfeatures,
          verbose = verbose
        )
        RunFastMNN(
          object.list = SplitObject(prepared, split.by = batch_col),
          assay = assay,
          features = VariableFeatures(prepared, assay = assay),
          reduction.name = "mnn",
          reconstructed.assay = "mnn.reconstructed",
          d = npcs,
          k = k_mnn,
          verbose = verbose
        )
      }
    ),
    cca = list(
      reduction = "integrated.cca",
      runner = function(object, batch_col, assay, nfeatures, npcs, k_mnn, k_weight, verbose) {
        prepared <- prepare_seurat_for_integration(
          object = object,
          batch_col = batch_col,
          assay = assay,
          nfeatures = nfeatures,
          npcs = npcs,
          verbose = verbose
        )
        IntegrateLayers(
          object = prepared,
          method = CCAIntegration,
          assay = assay,
          orig.reduction = "pca",
          new.reduction = "integrated.cca",
          features = VariableFeatures(prepared, assay = assay),
          dims = seq_len(npcs),
          k.weight = k_weight,
          verbose = verbose
        )
      }
    ),
    rpca = list(
      reduction = "integrated.rpca",
      runner = function(object, batch_col, assay, nfeatures, npcs, k_mnn, k_weight, verbose) {
        prepared <- prepare_seurat_for_integration(
          object = object,
          batch_col = batch_col,
          assay = assay,
          nfeatures = nfeatures,
          npcs = npcs,
          verbose = verbose
        )
        IntegrateLayers(
          object = prepared,
          method = RPCAIntegration,
          assay = assay,
          orig.reduction = "pca",
          new.reduction = "integrated.rpca",
          features = VariableFeatures(prepared, assay = assay),
          dims = seq_len(npcs),
          k.weight = k_weight,
          verbose = verbose
        )
      }
    ),
    jpca = list(
      reduction = "integrated.jpca",
      runner = function(object, batch_col, assay, nfeatures, npcs, k_mnn, k_weight, verbose) {
        prepared <- prepare_seurat_for_integration(
          object = object,
          batch_col = batch_col,
          assay = assay,
          nfeatures = nfeatures,
          npcs = npcs,
          verbose = verbose
        )
        IntegrateLayers(
          object = prepared,
          method = JointPCAIntegration,
          assay = assay,
          orig.reduction = "pca",
          new.reduction = "integrated.jpca",
          features = VariableFeatures(prepared, assay = assay),
          dims = seq_len(npcs),
          k.weight = k_weight,
          verbose = verbose
        )
      }
    )
  )

  if (requireNamespace("harmony", quietly = TRUE)) {
    methods$harmony <- list(
      reduction = "harmony",
      runner = function(object, batch_col, assay, nfeatures, npcs, k_mnn, k_weight, verbose) {
        prepared <- prepare_seurat_for_integration(
          object = object,
          batch_col = batch_col,
          assay = assay,
          nfeatures = nfeatures,
          npcs = npcs,
          verbose = verbose
        )
        IntegrateLayers(
          object = prepared,
          method = HarmonyIntegration,
          assay = assay,
          orig.reduction = "pca",
          new.reduction = "harmony",
          verbose = verbose
        )
      }
    )
  }

  if (requireNamespace("conos", quietly = TRUE)) {
    methods$conos <- list(
      reduction = "largeVis",
      runner = function(object, batch_col, assay, nfeatures, npcs, k_mnn, k_weight, verbose) {
        # Adapted from the SeuratWrappers Conos vignette: preprocess each batch
        # with Seurat, then build/embed a joint Conos graph and convert back.
        object.list <- prepare_seurat_panel_for_conos(
          object = object,
          batch_col = batch_col,
          assay = assay,
          nfeatures = nfeatures,
          npcs = npcs,
          verbose = verbose
        )

        conos_object <- conos::Conos$new(
          x = object.list,
          n.cores = 1L,
          verbose = verbose
        )
        conos_object$buildGraph(
          k = 15,
          k.self = 5,
          space = "PCA",
          ncomps = npcs,
          n.odgenes = nfeatures,
          matching.method = "mNN",
          metric = "angular",
          score.component.variance = TRUE,
          verbose = verbose
        )
        conos_object$findCommunities()
        conos_object$embedGraph(
          method = "largeVis",
          embedding.name = "largeVis",
          target.dims = npcs,
          verbose = verbose
        )

        as.Seurat(
          x = conos_object,
          reduction = "largeVis",
          verbose = verbose
        )
      }
    )
  }

  methods
}

resolve_batch_methods <- function(methods) {
  registry <- available_r_batch_methods()
  if (length(methods) == 1 && identical(tolower(methods), "all")) {
    return(registry)
  }

  requested <- tolower(methods)
  missing_methods <- setdiff(requested, names(registry))
  if (length(missing_methods) > 0) {
    stop(
      sprintf(
        "Unsupported methods: %s. Available methods: %s",
        paste(missing_methods, collapse = ", "),
        paste(names(registry), collapse = ", ")
      )
    )
  }

  registry[requested]
}

run_batch_correction_methods <- function(
  object,
  batch_col,
  methods = c("fastmnn", "cca", "rpca", "jpca"),
  assay = "RNA",
  output_dir = dirname(getwd()),
  output_prefix = "dataset",
  nfeatures = 2000,
  npcs = 50,
  k_mnn = 20,
  k_weight = 20,
  future_strategy = c("sequential", "current"),
  future_globals_max_size = NULL,
  verbose = FALSE
) {
  future_strategy <- match.arg(future_strategy)
  validate_batch_column(object, batch_col)
  resolved_methods <- resolve_batch_methods(methods)
  results <- vector("list", length(resolved_methods))

  for (index in seq_along(resolved_methods)) {
    method_name <- names(resolved_methods)[index]
    method_spec <- resolved_methods[[index]]
    corrected <- run_with_future_strategy(
      fun = function() {
        method_spec$runner(
          object = object,
          batch_col = batch_col,
          assay = assay,
          nfeatures = nfeatures,
          npcs = npcs,
          k_mnn = k_mnn,
          k_weight = k_weight,
          verbose = verbose
        )
      },
      future_strategy = future_strategy,
      future_globals_max_size = future_globals_max_size
    )

    embedding <- Embeddings(corrected, reduction = method_spec$reduction)
    save_file <- file.path(
      output_dir,
      sprintf("%s_%s_corrected.npy", output_prefix, method_name)
    )
    save_embedding_npy(embedding, save_file)

    results[[index]] <- data.frame(
      method = method_name,
      reduction = method_spec$reduction,
      output_file = save_file,
      n_cells = nrow(embedding),
      n_dims = ncol(embedding),
      stringsAsFactors = FALSE
    )
  }

  do.call(rbind, results)
}

run_h5ad_batch_correction_pipeline <- function(
  h5ad_file,
  batch_col,
  methods = c("fastmnn", "cca", "rpca", "jpca"),
  counts_path = "layers/counts",
  assay = "RNA",
  output_dir = dirname(h5ad_file),
  output_prefix = tools::file_path_sans_ext(basename(h5ad_file)),
  nfeatures = 2000,
  npcs = 50,
  k_mnn = 20,
  k_weight = 20,
  future_strategy = c("sequential", "current"),
  future_globals_max_size = NULL,
  save_rds = TRUE,
  rds_file = file.path(output_dir, sprintf("%s.rds", output_prefix)),
  verbose = FALSE
) {
  loaded <- make_seurat_from_h5ad(
    file = h5ad_file,
    counts_path = counts_path,
    assay = assay
  )

  if (save_rds) {
    dir.create(dirname(rds_file), recursive = TRUE, showWarnings = FALSE)
    saveRDS(loaded$object, file = rds_file)
  }

  outputs <- run_batch_correction_methods(
    object = loaded$object,
    batch_col = batch_col,
    methods = methods,
    assay = assay,
    output_dir = output_dir,
    output_prefix = output_prefix,
    nfeatures = nfeatures,
    npcs = npcs,
    k_mnn = k_mnn,
    k_weight = k_weight,
    future_strategy = future_strategy,
    future_globals_max_size = future_globals_max_size,
    verbose = verbose
  )

  list(
    object = loaded$object,
    obs = loaded$obs,
    var = loaded$var,
    rds_file = if (save_rds) rds_file else NULL,
    outputs = outputs
  )
}
