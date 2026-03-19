library(CoPro)
library(jsonlite)
library(reticulate)
library(yaml)

usage <- function() {
  stop("Usage: Rscript coprogression/coprogression.R <cell_type_a> <cell_type_b>", call. = FALSE)
}

script_path <- function() {
  file_arg <- "--file="
  match <- grep(file_arg, commandArgs(FALSE), value = TRUE)
  if (length(match) == 0) {
    stop("Could not determine script path.")
  }
  normalizePath(sub(file_arg, "", match[1], fixed = TRUE), mustWork = TRUE)
}

resolve_paths <- function() {
  root_dir <- dirname(dirname(script_path()))
  config_path <- normalizePath(file.path(root_dir, "config.yaml"), mustWork = TRUE)
  cfg <- read_yaml(config_path)
  processed_zarr <- normalizePath(file.path(dirname(config_path), cfg$output_directory, "processed", "processed.zarr"), mustWork = FALSE)
  analysis_dir <- normalizePath(file.path(dirname(config_path), cfg$output_directory, "analysis"), mustWork = FALSE)
  output_dir <- file.path(analysis_dir, "coprogression")
  list(
    config_path = config_path,
    processed_zarr = processed_zarr,
    analysis_dir = analysis_dir,
    output_dir = output_dir
  )
}

configure_python <- function(root_dir) {
  if (nzchar(Sys.getenv("RETICULATE_PYTHON", ""))) {
    use_python(Sys.getenv("RETICULATE_PYTHON"), required = TRUE)
    return(py_config()$python)
  }

  venv_dir <- file.path(root_dir, ".venv")
  if (file.exists(file.path(venv_dir, "pyvenv.cfg"))) {
    use_virtualenv(venv_dir, required = TRUE)
    return(py_config()$python)
  }

  py_config()$python
}

load_inputs <- function(processed_zarr) {
  tryCatch(import("spatialdata"), error = function(e) {
    stop("Python module 'spatialdata' is not available to reticulate: ", conditionMessage(e))
  })

  py_run_string(
    "
import numpy as np
import scipy.sparse as sp
from spatialdata import read_zarr

def extract_inputs(processed_zarr):
    sdata = read_zarr(processed_zarr)
    adata = sdata.tables['table']
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    if 'log_normalized' in adata.layers:
        matrix = adata.layers['log_normalized']
        layer = 'log_normalized'
    else:
        matrix = adata.X
        layer = 'X'
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    else:
        matrix = np.asarray(matrix)
    if {'x', 'y'}.issubset(obs.columns):
        coords = obs[['x', 'y']].to_numpy(dtype=float)
        coord_source = 'obs[x,y]'
    elif {'center_x', 'center_y'}.issubset(obs.columns):
        coords = obs[['center_x', 'center_y']].to_numpy(dtype=float)
        coord_source = 'obs[center_x,center_y]'
    elif 'spatial' in adata.obsm:
        coords = np.asarray(adata.obsm['spatial'])[:, :2]
        coord_source = 'obsm[spatial]'
    else:
        raise ValueError('No coordinates found in obs or obsm.')
    return {
        'matrix': matrix,
        'obs': obs,
        'coords': coords,
        'genes': adata.var_names.astype(str).tolist(),
        'cells': obs.index.tolist(),
        'layer': layer,
        'coord_source': coord_source,
    }
"
  )

  py_to_r(py$extract_inputs(processed_zarr))
}

build_copro_frame <- function(raw) {
  if (!"cell_type" %in% colnames(raw$obs)) {
    stop("processed.zarr table is missing obs$cell_type. Run main.py first.")
  }

  matrix <- as.matrix(raw$matrix)
  rownames(matrix) <- raw$cells
  colnames(matrix) <- raw$genes

  metadata <- raw$obs
  rownames(metadata) <- raw$cells
  metadata <- metadata[raw$cells, , drop = FALSE]

  coords <- as.matrix(raw$coords)
  location <- data.frame(
    x = as.numeric(coords[, 1]),
    y = as.numeric(coords[, 2]),
    row.names = raw$cells
  )

  list(
    normalized_data = matrix,
    location_data = location,
    metadata = metadata,
    cell_types = as.character(metadata$cell_type),
    used_layer = raw$layer,
    coord_source = raw$coord_source
  )
}

run_copro_pipeline <- function(object) {
  sigma_sets <- list(
    c(0.1, 0.14, 0.2, 0.5),
    c(1, 5, 10, 20, 50),
    c(100, 200, 500, 1000, 2000)
  )

  last_error <- NULL
  for (sigma_values in sigma_sets) {
    attempted <- computeKernelMatrix(object, sigmaValues = sigma_values, verbose = FALSE)
    attempted <- tryCatch(
      runSkrCCA(attempted, scalePCs = TRUE, nCC = 2, maxIter = 500),
      error = function(e) {
        last_error <<- conditionMessage(e)
        NULL
      }
    )
    if (is.null(attempted)) {
      next
    }
    attempted <- computeNormalizedCorrelation(attempted)
    attempted <- computeGeneAndCellScores(attempted)
    return(list(object = attempted, sigma_values = sigma_values))
  }

  stop("CoPro failed for all sigma sets. Last error: ", last_error)
}

resolve_requested_cell_types <- function(cell_types, cell_type_a, cell_type_b) {
  if (missing(cell_type_a) || missing(cell_type_b) || !nzchar(cell_type_a) || !nzchar(cell_type_b)) {
    usage()
  }
  if (identical(cell_type_a, cell_type_b)) {
    stop("Cell types must be distinct.")
  }

  available <- sort(unique(cell_types))
  aliases <- list(
    "monocyte/macrophage" = c("monocyte", "macrophage"),
    "prostate epithelial cell" = c("epithelial")
  )
  normalized_available <- tolower(available)
  requested <- c(cell_type_a, cell_type_b)
  remapped <- cell_types
  mapping <- vector("list", length(requested))
  names(mapping) <- requested

  for (requested_label in requested) {
    exact <- available[available == requested_label]
    if (length(exact) == 1) {
      mapping[[requested_label]] <- exact
      next
    }

    alias_key <- tolower(requested_label)
    keywords <- aliases[[alias_key]]
    if (is.null(keywords)) {
      preview <- paste(utils::head(available, 20), collapse = ", ")
      stop("Unknown cell type: ", requested_label, ". Available labels: ", preview)
    }

    matches <- available[Reduce(`|`, lapply(keywords, function(keyword) grepl(keyword, normalized_available, fixed = TRUE)))]
    if (length(matches) == 0) {
      preview <- paste(utils::head(available, 20), collapse = ", ")
      stop("Unknown cell type: ", requested_label, ". Available labels: ", preview)
    }

    remapped[cell_types %in% matches] <- requested_label
    mapping[[requested_label]] <- matches
  }

  if (length(intersect(mapping[[cell_type_a]], mapping[[cell_type_b]])) > 0) {
    stop("Resolved cell type groups overlap; choose more specific labels.")
  }

  counts <- table(remapped)
  list(
    remapped_cell_types = remapped,
    counts = setNames(as.integer(counts[c(cell_type_a, cell_type_b)]), c(cell_type_a, cell_type_b)),
    mapping = mapping
  )
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 2) {
    usage()
  }

  cell_type_a <- args[1]
  cell_type_b <- args[2]
  paths <- resolve_paths()
  if (!dir.exists(paths$analysis_dir)) {
    stop("Analysis directory does not exist: ", paths$analysis_dir)
  }
  if (!dir.exists(paths$processed_zarr)) {
    stop("processed.zarr not found: ", paths$processed_zarr)
  }
  dir.create(paths$output_dir, recursive = TRUE, showWarnings = FALSE)

  python_path <- configure_python(dirname(dirname(script_path())))
  raw <- load_inputs(paths$processed_zarr)
  copro <- build_copro_frame(raw)
  resolved <- resolve_requested_cell_types(copro$cell_types, cell_type_a, cell_type_b)

  object <- newCoProSingle(
    normalizedData = copro$normalized_data,
    locationData = copro$location_data,
    metaData = copro$metadata,
    cellTypes = resolved$remapped_cell_types
  )
  object <- subsetData(object, cellTypesOfInterest = c(cell_type_a, cell_type_b), saveOriginal = TRUE)
  object <- computePCA(object, nPCA = 40, center = TRUE, scale. = TRUE)
  object <- computeDistance(object, distType = "Euclidean2D", normalizeDistance = FALSE, verbose = FALSE)
  fitted <- run_copro_pipeline(object)
  object <- fitted$object

  sigma_choice <- as.numeric(object@sigmaValueChoice)
  if (length(sigma_choice) != 1 || is.na(sigma_choice)) {
    stop("CoPro did not produce a single sigmaValueChoice.")
  }

  norm_corr <- getNormCorr(object)
  corr_two_types <- getCorrTwoTypes(
    object,
    sigmaValueChoice = sigma_choice,
    cellTypeA = cell_type_a,
    cellTypeB = cell_type_b,
    ccIndex = 1
  )
  cell_scores <- getCellScoresInSitu(object, sigmaValueChoice = sigma_choice, ccIndex = 1)

  write.csv(norm_corr, file.path(paths$output_dir, "copro_norm_corr.csv"), row.names = FALSE)
  write.csv(corr_two_types, file.path(paths$output_dir, "copro_corr_two_types.csv"), row.names = FALSE)
  write.csv(cell_scores, file.path(paths$output_dir, "copro_cell_scores_insitu.csv"), row.names = FALSE)

  summary <- list(
    config_path = paths$config_path,
    processed_zarr = paths$processed_zarr,
    analysis_dir = paths$analysis_dir,
    output_dir = paths$output_dir,
    python = python_path,
    used_layer = copro$used_layer,
    coordinate_source = copro$coord_source,
    selected_cell_types = c(cell_type_a, cell_type_b),
    selected_cell_type_counts = as.list(resolved$counts),
    resolved_labels = resolved$mapping,
    sigma_values = fitted$sigma_values,
    sigma_value_choice = sigma_choice,
    n_pca = 40,
    n_cc = 2,
    max_iter = 500,
    normalize_distance = FALSE,
    rows = list(
      normalized_correlation = nrow(norm_corr),
      corr_two_types = nrow(corr_two_types),
      cell_scores_insitu = nrow(cell_scores)
    ),
    outputs = list(
      normalized_correlation = "copro_norm_corr.csv",
      corr_two_types = "copro_corr_two_types.csv",
      cell_scores_insitu = "copro_cell_scores_insitu.csv"
    )
  )
  write_json(summary, file.path(paths$output_dir, "copro_summary.json"), pretty = TRUE, auto_unbox = TRUE)

  message("Wrote CoPro outputs to ", paths$output_dir)
}

main()
