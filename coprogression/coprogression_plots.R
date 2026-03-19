library(ggplot2)
library(jsonlite)
library(yaml)

usage <- function() {
  stop("Usage: Rscript coprogression/coprogression_plots.R <cell_type_a> <cell_type_b>", call. = FALSE)
}

script_path <- function() {
  file_arg <- "--file="
  match <- grep(file_arg, commandArgs(FALSE), value = TRUE)
  if (length(match) == 0) {
    stop("Could not determine script path.")
  }
  normalizePath(sub(file_arg, "", match[1], fixed = TRUE), mustWork = TRUE)
}

resolve_output_dir <- function() {
  root_dir <- dirname(dirname(script_path()))
  config_path <- normalizePath(file.path(root_dir, "config.yaml"), mustWork = TRUE)
  cfg <- read_yaml(config_path)
  normalizePath(file.path(dirname(config_path), cfg$output_directory, "analysis", "coprogression"), mustWork = FALSE)
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 2) {
    usage()
  }

  cell_type_a <- args[1]
  cell_type_b <- args[2]
  if (identical(cell_type_a, cell_type_b)) {
    stop("Cell types must be distinct.")
  }

  output_dir <- resolve_output_dir()
  summary_path <- file.path(output_dir, "copro_summary.json")
  if (!file.exists(summary_path)) {
    stop("Missing copro_summary.json in ", output_dir, ". Run coprogression.R first.")
  }

  summary <- read_json(summary_path, simplifyVector = TRUE)
  if (!setequal(summary$selected_cell_types, c(cell_type_a, cell_type_b))) {
    stop("copro_summary.json was generated for different cell types. Re-run coprogression.R first.")
  }

  norm_corr <- read.csv(file.path(output_dir, "copro_norm_corr.csv"), stringsAsFactors = FALSE)
  corr_two_types <- read.csv(file.path(output_dir, "copro_corr_two_types.csv"), stringsAsFactors = FALSE)
  cell_scores <- read.csv(file.path(output_dir, "copro_cell_scores_insitu.csv"), stringsAsFactors = FALSE)

  norm_plot <- ggplot(data = norm_corr, aes(x = sigmaValues, y = normalizedCorrelation, group = 1)) +
    geom_point() +
    geom_line() +
    facet_wrap(vars(ct12, CC_index)) +
    xlab("Sigma squared") +
    ylab("Norm. Corr.") +
    ggtitle("Norm. Corr. across sigma squared values") +
    theme_minimal()

  corr_plot <- ggplot(corr_two_types) +
    geom_point(aes(x = AK, y = B)) +
    ggtitle(paste("Correlation plot between", cell_type_a, "and", cell_type_b)) +
    xlab(paste(cell_type_a, "%*% Kernal_AB")) +
    ylab(cell_type_b) +
    theme_minimal()

  binary_plot <- ggplot(data = cell_scores) +
    geom_point(aes(x = x, y = y, color = cellScores_b), size = 0.8) +
    coord_fixed() +
    theme_minimal()

  continuous_plot <- ggplot(data = cell_scores) +
    geom_point(aes(x = x, y = y, color = cellScores), size = 0.8) +
    coord_fixed() +
    theme_minimal()

  ggsave(file.path(output_dir, "copro_norm_corr.png"), norm_plot, width = 9, height = 5, dpi = 200)
  ggsave(file.path(output_dir, "copro_corr_two_types.png"), corr_plot, width = 7, height = 5, dpi = 200)
  ggsave(file.path(output_dir, "copro_cell_scores_binary.png"), binary_plot, width = 7, height = 6, dpi = 200)
  ggsave(file.path(output_dir, "copro_cell_scores_continuous.png"), continuous_plot, width = 7, height = 6, dpi = 200)

  message("Wrote CoPro plots to ", output_dir)
}

main()
