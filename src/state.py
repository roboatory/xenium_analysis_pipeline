from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Configuration


def build_state_path(configuration: Configuration) -> Path:
    """Return the path for provenance state output."""

    return configuration.results_directory / "state.json"


def configuration_settings_snapshot(configuration: Configuration) -> dict[str, Any]:
    """Build a serializable snapshot of config settings used for this run."""

    pipeline = configuration.pipeline
    plots = configuration.plots
    return {
        "raw_data_directory": str(configuration.raw_data_directory),
        "output_directory": str(configuration.output_directory),
        "annotation_model": configuration.annotation_model,
        "pipeline": {
            "minimum_counts": pipeline.minimum_counts,
            "maximum_counts_quantile": pipeline.maximum_counts_quantile,
            "minimum_cells": pipeline.minimum_cells,
            "pca_n_components": pipeline.pca_n_components,
            "neighborhood_radius": pipeline.neighborhood_radius,
            "colocalization_radius": pipeline.colocalization_radius,
            "colocalization_number_of_permutations": (
                pipeline.colocalization_number_of_permutations
            ),
            "colocalization_minimum_cells": pipeline.colocalization_minimum_cells,
            "domain_n_clusters": pipeline.domain_n_clusters,
            "rank_top_n": pipeline.rank_top_n,
            "minimum_logarithm_fold_change": pipeline.minimum_logarithm_fold_change,
            "maximum_adjusted_p_value": pipeline.maximum_adjusted_p_value,
        },
        "plots": {
            "genes_to_plot": list(plots.genes_to_plot),
        },
    }


def save_state(path: Path, state_payload: dict[str, Any]) -> None:
    """Write run configuration snapshot JSON."""

    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(state_payload, file_handle, indent=2, sort_keys=True)
