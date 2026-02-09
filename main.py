from __future__ import annotations

import argparse
from dataclasses import dataclass
import dask.config

dask.config.set({"dataframe.query-planning": True})

import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import pandas as pd  # noqa: E402

from src import analysis, annotation, colocalization, io, plotting, preprocessing  # noqa: E402
from src.config import Config  # noqa: E402


@dataclass(frozen=True)
class StagePaths:
    processed_path: Path
    enriched_genes_path: Path
    cluster_labels_path: Path
    annotations_path: Path
    domain_labels_path: Path
    domain_annotations_path: Path
    state_path: Path


def parse_arguments() -> argparse.Namespace:
    """Parse CLI controls for forcing selective reruns."""

    parser = argparse.ArgumentParser(description="Run the xenium analysis pipeline.")
    parser.add_argument(
        "--force-process",
        action="store_true",
        help="Force rerun of preprocessing/clustering and rewrite processed.zarr.",
    )
    parser.add_argument(
        "--force-annotate",
        action="store_true",
        help="Force rerun of cluster annotation and rewrite processed.zarr.",
    )
    parser.add_argument(
        "--force-colocalization",
        action="store_true",
        help="Force rerun of colocalization analysis and rewrite processed.zarr.",
    )
    return parser.parse_args()


def _hash_payload(payload: object) -> str:
    """Return deterministic SHA256 for a JSON-serializable payload."""

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _process_signature(configuration: Config) -> str:
    """Create the signature for ingestion/preprocessing/clustering stage."""

    pipeline = configuration.pipeline
    payload = {
        "raw_data_directory": str(configuration.raw_data_directory),
        "minimum_counts": pipeline.minimum_counts,
        "maximum_counts_quantile": pipeline.maximum_counts_quantile,
        "minimum_cells": pipeline.minimum_cells,
        "n_top_genes": pipeline.n_top_genes,
        "n_components": pipeline.n_components,
        "leiden_resolution": pipeline.leiden_resolution,
        "rank_top_n": pipeline.rank_top_n,
        "minimum_logarithm_fold_change": pipeline.minimum_logarithm_fold_change,
        "maximum_adjusted_p_value": pipeline.maximum_adjusted_p_value,
    }
    return _hash_payload(payload)


def _annotate_signature(configuration: Config) -> str:
    """Create the signature for cluster annotation stage."""

    payload = {"annotation_model": configuration.annotation_model}
    return _hash_payload(payload)


def _colocalization_signature(configuration: Config) -> str:
    """Create the signature for colocalization stage."""

    pipeline = configuration.pipeline
    payload = {
        "neighborhood_radius": pipeline.neighborhood_radius,
        "domain_n_clusters": pipeline.domain_n_clusters,
    }
    return _hash_payload(payload)


def _load_state(path: Path) -> dict[str, Any]:
    """Load state file; return empty dict when missing or invalid."""

    try:
        with path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items()}


def _save_state(path: Path, state_payload: dict[str, Any]) -> None:
    """Write state JSON."""

    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(state_payload, file_handle, indent=2, sort_keys=True)


def _configuration_settings_snapshot(configuration: Config) -> dict[str, Any]:
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
            "n_top_genes": pipeline.n_top_genes,
            "n_components": pipeline.n_components,
            "leiden_resolution": pipeline.leiden_resolution,
            "rank_top_n": pipeline.rank_top_n,
            "minimum_logarithm_fold_change": pipeline.minimum_logarithm_fold_change,
            "maximum_adjusted_p_value": pipeline.maximum_adjusted_p_value,
            "neighborhood_radius": pipeline.neighborhood_radius,
            "domain_n_clusters": pipeline.domain_n_clusters,
        },
        "plots": {
            "plot_boundaries": plots.plot_boundaries,
            "plot_transcripts": plots.plot_transcripts,
            "genes_to_plot": list(plots.genes_to_plot),
        },
    }


def _load_configuration() -> Config:
    """Load and initialize top-level configuration."""

    configuration = Config()
    configuration.load_from_yaml(Path("config.yaml").resolve())
    configuration.create_directories()
    return configuration


def _build_stage_paths(configuration: Config) -> StagePaths:
    """Build canonical input/output paths used for stage decisions."""

    return StagePaths(
        processed_path=configuration.processed_data_directory / "processed.zarr",
        enriched_genes_path=configuration.results_directory
        / "cluster_enriched_genes.json",
        cluster_labels_path=configuration.results_directory / "leiden_clusters.csv",
        annotations_path=configuration.results_directory / "cluster_annotations.json",
        domain_labels_path=configuration.results_directory
        / "spatial_domain_labels.csv",
        domain_annotations_path=configuration.results_directory
        / "spatial_domain_annotations.json",
        state_path=configuration.results_directory / "state.json",
    )


def _should_run_process_stage(
    arguments: argparse.Namespace,
    paths: StagePaths,
    state_cache: dict[str, Any],
    process_signature: str,
) -> bool:
    """Determine whether ingestion/preprocessing stage must rerun."""

    return (
        arguments.force_process
        or not paths.processed_path.exists()
        or not paths.enriched_genes_path.exists()
        or not paths.cluster_labels_path.exists()
        or state_cache.get("process_signature") != process_signature
    )


def _should_run_annotate_stage(
    arguments: argparse.Namespace,
    paths: StagePaths,
    state_cache: dict[str, Any],
    should_run_process_stage: bool,
    process_signature: str,
    annotate_signature: str,
) -> bool:
    """Determine whether annotation stage must rerun."""

    return (
        arguments.force_annotate
        or should_run_process_stage
        or not paths.annotations_path.exists()
        or state_cache.get("annotate_signature") != annotate_signature
        or state_cache.get("annotate_dependency_process_signature") != process_signature
    )


def _should_run_colocalization_stage(
    arguments: argparse.Namespace,
    paths: StagePaths,
    state_cache: dict[str, Any],
    should_run_process_stage: bool,
    should_run_annotate_stage: bool,
    annotate_signature: str,
    colocalization_signature: str,
) -> bool:
    """Determine whether colocalization stage must rerun."""

    return (
        arguments.force_colocalization
        or should_run_process_stage
        or should_run_annotate_stage
        or not paths.domain_labels_path.exists()
        or not paths.domain_annotations_path.exists()
        or state_cache.get("colocalization_signature") != colocalization_signature
        or state_cache.get("colocalization_dependency_annotate_signature")
        != annotate_signature
    )


def _run_ingestion_preprocessing_stage(configuration: Config) -> None:
    """Run ingestion -> preprocessing -> clustering -> marker ranking stage."""

    spatial_data = io.load_xenium(configuration)
    annotated_data = spatial_data["table"]
    if "morphology_focus" in spatial_data:
        del spatial_data["morphology_focus"]
        io.write_spatialdata_zarr(spatial_data, annotated_data, configuration)

    spatial_data = io.read_spatialdata_zarr(configuration)
    annotated_data = spatial_data["table"]

    if configuration.plots.plot_boundaries:
        plotting.plot_cell_and_nucleus_boundaries(spatial_data, configuration)
    if configuration.plots.plot_transcripts:
        plotting.plot_transcripts(
            spatial_data,
            configuration,
            list(configuration.plots.genes_to_plot),
            ["blue", "orange"],
        )

    annotated_data.layers["counts"] = annotated_data.X.copy()

    plotting.plot_qc_histogram(
        annotated_data,
        [
            configuration.pipeline.minimum_counts,
            configuration.pipeline.maximum_counts_quantile,
        ],
        ["crimson", "goldenrod"],
        configuration,
    )
    preprocessing.filter_cells_and_genes(
        annotated_data,
        configuration.pipeline.minimum_counts,
        configuration.pipeline.maximum_counts_quantile,
        configuration.pipeline.minimum_cells,
    )
    preprocessing.normalize_and_scale(
        annotated_data, configuration.pipeline.n_top_genes
    )

    analysis.run_clustering(
        annotated_data,
        configuration.pipeline.n_components,
        configuration.pipeline.leiden_resolution,
    )
    analysis.run_umap(annotated_data)
    analysis.rank_genes(annotated_data)
    enriched_gene_lists = analysis.compute_enriched_genes(
        annotated_data,
        pd.unique(annotated_data.obs["leiden"]),
        configuration.pipeline.rank_top_n,
        configuration.pipeline.minimum_logarithm_fold_change,
        configuration.pipeline.maximum_adjusted_p_value,
    )
    plotting.plot_rank_genes_dotplot(annotated_data, configuration, n_genes=5)

    io.write_cluster_labels(annotated_data, configuration)
    io.write_analysis_artifact(
        configuration,
        "enriched_genes",
        enriched_genes=enriched_gene_lists,
    )

    spatial_data["table"] = annotated_data
    io.write_spatialdata_zarr(spatial_data, annotated_data, configuration)


def _run_annotation_stage(configuration: Config) -> None:
    """Run LLM-driven cell-type annotation and persist updated zarr."""

    spatial_data = io.read_spatialdata_zarr(configuration)
    annotated_data = spatial_data["table"]

    enriched_gene_lists = io.load_enriched_genes(configuration)
    cluster_annotations = annotation.annotate_clusters_with_llm(
        annotation_evidence_by_cluster=enriched_gene_lists,
        model=configuration.annotation_model,
        evidence_type="marker_genes",
    )
    io.write_annotations(cluster_annotations, configuration, "cluster")

    annotated_data.obs["cell_type"] = (
        annotated_data.obs["leiden"]
        .astype(str)
        .map(
            {
                cluster: value["cell_type"]
                for cluster, value in cluster_annotations.items()
            }
        )
    )

    spatial_data["table"] = annotated_data
    io.write_spatialdata_zarr(spatial_data, annotated_data, configuration)


def _run_colocalization_stage(configuration: Config) -> None:
    """Run colocalization/neighborhood analysis and persist updated zarr."""

    spatial_data = io.read_spatialdata_zarr(configuration)
    annotated_data = spatial_data["table"]

    colocalization.compute_neighborhood_composition(
        annotated_data, configuration.pipeline.neighborhood_radius
    )
    colocalization.assign_spatial_domains(
        annotated_data, configuration.pipeline.domain_n_clusters
    )
    domain_signatures = analysis.build_domain_signatures(annotated_data)
    domain_annotations = annotation.annotate_clusters_with_llm(
        annotation_evidence_by_cluster=domain_signatures,
        model=configuration.annotation_model,
        evidence_type="neighborhood_cell_types",
    )

    io.write_annotations(domain_annotations, configuration, "domain")
    annotated_data.obs["spatial_domain_label"] = (
        annotated_data.obs["spatial_domain"]
        .astype(str)
        .map(
            {
                domain_id: value["cell_type"]
                for domain_id, value in domain_annotations.items()
            }
        )
    )

    io.write_analysis_artifact(
        configuration,
        "spatial_domains",
        annotated_data=annotated_data,
        domain_key="spatial_domain",
    )

    spatial_data["table"] = annotated_data
    io.write_spatialdata_zarr(spatial_data, annotated_data, configuration)


def main() -> None:
    arguments = parse_arguments()
    configuration = _load_configuration()
    paths = _build_stage_paths(configuration)
    run_configuration_settings = _configuration_settings_snapshot(configuration)

    state_cache = _load_state(paths.state_path)
    process_signature = _process_signature(configuration)
    annotate_signature = _annotate_signature(configuration)
    colocalization_signature = _colocalization_signature(configuration)

    should_run_process_stage = _should_run_process_stage(
        arguments,
        paths,
        state_cache,
        process_signature,
    )
    if should_run_process_stage:
        _run_ingestion_preprocessing_stage(configuration)
        state_cache = {
            "process_signature": process_signature,
            "annotate_signature": "",
            "annotate_dependency_process_signature": "",
            "colocalization_signature": "",
            "colocalization_dependency_annotate_signature": "",
        }
        state_cache["run_configuration_settings"] = run_configuration_settings
        _save_state(paths.state_path, state_cache)

    should_run_annotate_stage = _should_run_annotate_stage(
        arguments,
        paths,
        state_cache,
        should_run_process_stage=should_run_process_stage,
        process_signature=process_signature,
        annotate_signature=annotate_signature,
    )
    if should_run_annotate_stage:
        _run_annotation_stage(configuration)
        state_cache["annotate_signature"] = annotate_signature
        state_cache["annotate_dependency_process_signature"] = process_signature
        state_cache["colocalization_signature"] = ""
        state_cache["colocalization_dependency_annotate_signature"] = ""
        state_cache["run_configuration_settings"] = run_configuration_settings
        _save_state(paths.state_path, state_cache)

    should_run_colocalization_stage = _should_run_colocalization_stage(
        arguments,
        paths,
        state_cache,
        should_run_process_stage=should_run_process_stage,
        should_run_annotate_stage=should_run_annotate_stage,
        annotate_signature=annotate_signature,
        colocalization_signature=colocalization_signature,
    )
    if should_run_colocalization_stage:
        _run_colocalization_stage(configuration)
        state_cache["colocalization_signature"] = colocalization_signature
        state_cache["colocalization_dependency_annotate_signature"] = annotate_signature
        state_cache["run_configuration_settings"] = run_configuration_settings
        _save_state(paths.state_path, state_cache)

    spatial_data = io.read_spatialdata_zarr(configuration)
    plotting.plot_umap_leiden(spatial_data, configuration, cluster_key="cell_type")
    plotting.plot_cluster_overlay(
        spatial_data,
        configuration,
        cluster_key="cell_type",
    )
    plotting.plot_cluster_overlay(
        spatial_data,
        configuration,
        cluster_key="spatial_domain_label",
    )


if __name__ == "__main__":
    main()
