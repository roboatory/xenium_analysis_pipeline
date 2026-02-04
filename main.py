from __future__ import annotations

import argparse
from dataclasses import asdict
import dask.config

dask.config.set({"dataframe.query-planning": True})

import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from src import analysis, annotation, io, plotting, preprocessing  # noqa: E402
from src.config import Config  # noqa: E402

ANNOTATION_MODEL = "llama3.1:8b"


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


def _hash_json_file(path: Path) -> str | None:
    """Hash parsed JSON contents for stable change detection."""

    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return None
    return _hash_payload(payload)


def _processed_config_hash(configuration: Config) -> str:
    """Create the hash used to decide if processing must rerun."""

    payload = {
        "raw_data_directory": str(configuration.raw_data_directory),
        "pipeline": asdict(configuration.pipeline),
    }
    return _hash_payload(payload)


def _load_state(path: Path) -> dict[str, str]:
    """Load state file; return empty dict when missing or invalid."""

    try:
        with path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _save_state(path: Path, state_payload: dict[str, str]) -> None:
    """Write state JSON."""

    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(state_payload, file_handle, indent=2, sort_keys=True)


def main() -> None:
    arguments = parse_arguments()

    configuration = Config()
    configuration.load_from_yaml(Path("config.yaml").resolve())
    configuration.create_directories()

    processed_path = configuration.processed_data_directory / "processed.zarr"
    enriched_genes_path = (
        configuration.results_directory / "cluster_enriched_genes.json"
    )
    cluster_labels_path = configuration.results_directory / "leiden_clusters.csv"
    annotations_path = configuration.results_directory / "cluster_annotations.json"
    state_path = configuration.results_directory / "state.json"

    state_cache = _load_state(state_path)
    config_hash = _processed_config_hash(configuration)
    should_run_process = (
        arguments.force_process
        or not processed_path.exists()
        or not enriched_genes_path.exists()
        or not cluster_labels_path.exists()
        or state_cache.get("processed_config_hash") != config_hash
    )

    if should_run_process:
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

        plotting.plot_umap_leiden(spatial_data, configuration)
        plotting.plot_cluster_overlay(spatial_data, configuration)

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
        io.write_enriched_genes(enriched_gene_lists, configuration)

        enriched_hash = _hash_json_file(enriched_genes_path)
        _save_state(
            state_path,
            {
                "processed_config_hash": config_hash,
                "enriched_genes_hash": enriched_hash or "",
                "annotations_hash": "",
                "annotation_model": ANNOTATION_MODEL,
            },
        )

        state_cache = _load_state(state_path)
    else:
        spatial_data = io.read_spatialdata_zarr(configuration)
        annotated_data = spatial_data["table"]

    enriched_hash = _hash_json_file(enriched_genes_path)
    annotations_hash = _hash_json_file(annotations_path)
    should_run_annotate = (
        arguments.force_annotate
        or should_run_process
        or not annotations_path.exists()
        or enriched_hash is None
        or state_cache.get("enriched_genes_hash") != enriched_hash
        or state_cache.get("annotations_hash") != (annotations_hash or "")
        or state_cache.get("annotation_model") != ANNOTATION_MODEL
    )

    if should_run_annotate:
        if not should_run_process:
            spatial_data = io.read_spatialdata_zarr(configuration)
            annotated_data = spatial_data["table"]
        enriched_gene_lists = io.load_enriched_genes(configuration)
        cluster_annotations = annotation.annotate_clusters_with_llm(
            enriched_gene_lists,
            model=ANNOTATION_MODEL,
        )
        io.write_cluster_annotations(cluster_annotations, configuration)

        annotated_data.obs["cell_type"] = (
            annotated_data.obs["leiden"]
            .astype(str)
            .map({c: v["cell_type"] for c, v in cluster_annotations.items()})
        )

        spatial_data["table"] = annotated_data
        io.write_spatialdata_zarr(spatial_data, annotated_data, configuration)
        annotations_hash = _hash_json_file(annotations_path)

    state_payload = {
        "processed_config_hash": config_hash,
        "enriched_genes_hash": enriched_hash or "",
        "annotations_hash": annotations_hash or "",
        "annotation_model": ANNOTATION_MODEL,
    }
    _save_state(state_path, state_payload)


if __name__ == "__main__":
    main()
