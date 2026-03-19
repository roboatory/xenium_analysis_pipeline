from __future__ import annotations

from pathlib import Path


from src import (
    analysis,
    annotation,
    colocalization,
    io,
    plotting,
    preprocessing,
    spatial_domains,
    state,
)
from src.config import Configuration

CONFIG_PATH = Path("config.yaml").resolve()


def load_configuration() -> Configuration:
    """Load and initialize top-level configuration."""

    configuration = Configuration()
    configuration.load_from_yaml(CONFIG_PATH)
    configuration.create_directories()
    return configuration


def run_preprocess_cluster_stage(configuration: Configuration) -> None:
    """Run preprocessing and clustering steps on pre-ingested data."""

    spatial_data = io.read_spatialdata_zarr(configuration)
    annotated_data = spatial_data["table"]

    plotting.plot_cell_and_nucleus_boundaries(configuration, spatial_data)
    plotting.plot_transcripts(
        configuration,
        spatial_data,
        list(configuration.plots.genes_to_plot),
        ["blue", "orange"],
    )

    if "counts" in annotated_data.layers:
        annotated_data.X = annotated_data.layers["counts"].copy()
    else:
        annotated_data.layers["counts"] = annotated_data.X.copy()

    plotting.plot_qc_histogram(
        configuration,
        annotated_data,
        [
            configuration.pipeline.minimum_counts,
            configuration.pipeline.maximum_counts_quantile,
        ],
        ["crimson", "goldenrod"],
    )
    preprocessing.filter_cells_and_genes(
        annotated_data,
        configuration.pipeline.minimum_counts,
        configuration.pipeline.maximum_counts_quantile,
        configuration.pipeline.minimum_cells,
    )
    preprocessing.normalize_and_scale(annotated_data)
    analysis.run_clustering(
        annotated_data,
        configuration.pipeline.pca_n_components,
    )
    analysis.run_umap(annotated_data)
    analysis.rank_genes(annotated_data)
    enriched_gene_lists = analysis.compute_enriched_genes(
        annotated_data,
        configuration.pipeline.rank_top_n,
        configuration.pipeline.minimum_logarithm_fold_change,
        configuration.pipeline.maximum_adjusted_p_value,
    )
    plotting.plot_rank_genes_dotplot(configuration, annotated_data)

    io.write_labels(configuration, annotated_data, "cluster")
    io.write_enriched_genes(configuration, enriched_gene_lists)

    spatial_data["table"] = annotated_data
    io.write_spatialdata_zarr(configuration, spatial_data)


def run_annotation_stage(configuration: Configuration) -> None:
    """Run LLM-driven cell-type annotation and persist updated zarr."""

    spatial_data = io.read_spatialdata_zarr(configuration)
    annotated_data = spatial_data["table"]

    enriched_gene_lists = io.read_enriched_genes(configuration)
    cluster_annotations = annotation.annotate_clusters_with_llm(
        enriched_gene_lists,
        configuration.annotation_model,
        "marker_genes",
    )
    io.write_annotations(configuration, cluster_annotations, "cluster")

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

    plotting.plot_umap_leiden(
        configuration,
        spatial_data,
    )
    plotting.plot_cluster_overlay(
        configuration,
        spatial_data,
        cluster_key="cell_type",
    )

    spatial_data["table"] = annotated_data
    io.write_spatialdata_zarr(configuration, spatial_data)


def run_neighborhood_stage(configuration: Configuration) -> None:
    """Run neighborhood analysis and persist updated zarr."""

    spatial_data = io.read_spatialdata_zarr(configuration)
    annotated_data = spatial_data["table"]

    spatial_domains.compute_neighborhood_composition(
        annotated_data, configuration.pipeline.neighborhood_colocalization_radius
    )
    spatial_domains.assign_spatial_domains(
        annotated_data, configuration.pipeline.domain_n_clusters
    )
    domain_signatures = spatial_domains.build_domain_signatures(annotated_data)
    domain_annotations = annotation.annotate_clusters_with_llm(
        domain_signatures,
        configuration.annotation_model,
        "neighborhood_cell_types",
    )

    io.write_annotations(configuration, domain_annotations, "domain")
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

    io.write_labels(configuration, annotated_data, "domain")

    plotting.plot_cluster_overlay(
        configuration,
        spatial_data,
        cluster_key="spatial_domain_label",
    )

    spatial_data["table"] = annotated_data
    io.write_spatialdata_zarr(configuration, spatial_data)


def run_colocalization_stage(configuration: Configuration) -> None:
    """Run observed cell-type contact colocalization and write artifacts."""

    spatial_data = io.read_spatialdata_zarr(configuration)
    annotated_data = spatial_data["table"]

    counts, row_proportions = colocalization.compute_observed_contact_matrices(
        annotated_data,
    )
    plotting.plot_colocalization_contact_counts(configuration, counts)
    plotting.plot_colocalization_contact_row_proportions(
        configuration,
        row_proportions,
    )

    permutation_results = colocalization.compute_permutation_significance(
        annotated_data,
        number_of_permutations=(
            configuration.pipeline.colocalization_number_of_permutations
        ),
        minimum_cells=configuration.pipeline.colocalization_minimum_cells,
    )
    plotting.plot_colocalization_log2_fold_enrichment(
        configuration,
        permutation_results["log2_fold_enrichment"],
    )
    plotting.plot_colocalization_log2_fold_enrichment_significant_only(
        configuration,
        permutation_results["log2_fold_enrichment"],
        permutation_results["significant_mask"],
    )


def main() -> None:
    configuration = load_configuration()
    ingested_path = configuration.processed_data_directory / "processed.zarr"
    if not ingested_path.exists():
        msg = (
            f"Ingested data not found at '{ingested_path}'. "
            "Run `uv run python3 ingest.py` before launching main.py."
        )
        raise FileNotFoundError(msg)

    run_preprocess_cluster_stage(configuration)
    run_annotation_stage(configuration)
    run_neighborhood_stage(configuration)
    run_colocalization_stage(configuration)

    state_path = state.build_state_path(configuration)
    configuration_snapshot = state.configuration_settings_snapshot(configuration)
    io.save_state(state_path, configuration_snapshot)


if __name__ == "__main__":
    main()
