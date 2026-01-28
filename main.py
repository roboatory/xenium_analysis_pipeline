from __future__ import annotations

from pathlib import Path
from typing import Any

import scanpy as sc
import yaml

from src import analysis, io, plotting, preprocessing
from src.config import Config


def _load_configuration(configuration_path: Path) -> dict[str, Any]:
    with configuration_path.open("r") as configuration_file:
        configuration_data = yaml.safe_load(configuration_file)
    return configuration_data if isinstance(configuration_data, dict) else {}


def main() -> None:
    configuration_path = Path(__file__).resolve().parent / "config.yaml"
    raw_configuration = _load_configuration(configuration_path)

    # Build a single configuration object that carries all paths and parameters.
    configuration = Config.from_dict(raw_configuration, config_dir=configuration_path.parent)
    configuration.ensure_dirs()

    pipeline_configuration = configuration.pipeline
    plots_configuration = configuration.plots

    preprocessing.configure_scanpy()
    # Prefer using configuration for input paths instead of passing data_directory directly.
    spatial_data = io.load_xenium(config=configuration)
    annotated_data = preprocessing.init_adata(spatial_data)

    if "total_counts" not in annotated_data.obs:
        sc.pp.calculate_qc_metrics(annotated_data, inplace=True)

    minimum_counts = pipeline_configuration.min_counts
    maximum_counts_quantile = pipeline_configuration.max_counts_quantile
    minimum_cells = pipeline_configuration.min_cells
    maximum_count_threshold = preprocessing.filter_cells_and_genes(
        annotated_data,
        min_counts=minimum_counts,
        max_counts_quantile=maximum_counts_quantile,
        min_cells=minimum_cells,
    )

    preprocessing.normalize_and_scale(
        annotated_data,
        n_top_genes=pipeline_configuration.n_top_genes,
    )

    analysis.run_clustering(
        annotated_data,
        n_comps=pipeline_configuration.n_comps,
        leiden_resolution=pipeline_configuration.leiden_resolution,
    )
    analysis.run_umap(annotated_data)
    analysis.rank_genes(annotated_data)

    ordered_clusters, _ = analysis.ordered_clusters(annotated_data.obs["leiden"].astype(str))
    enriched_gene_lists, enriched_gene_rows = analysis.compute_enriched_genes(
        annotated_data,
        clusters=ordered_clusters,
        top_n=pipeline_configuration.rank_top_n,
        min_logfc=pipeline_configuration.min_logfc,
        max_adj_pval=pipeline_configuration.max_adj_pval,
    )

    io.write_cluster_labels(annotated_data, config=configuration)
    io.write_enriched_genes(enriched_gene_lists, enriched_gene_rows, config=configuration)
    annotations_path = io.ensure_placeholder_annotations(None, ordered_clusters, config=configuration)
    annotations = io.read_json_if_exists(annotations_path)
    analysis.apply_celltype_annotations(annotated_data, annotations)

    io.write_spatialdata_zarr(spatial_data, annotated_data, config=configuration)

    plotting.plot_qc_hist(
        annotated_data,
        cutoffs=[minimum_counts, int(maximum_count_threshold)],
        cutoff_colors=["crimson", "goldenrod"],
        config=configuration,
    )
    plotting.plot_umap_leiden(annotated_data, config=configuration)
    plotting.plot_rank_genes_dotplot(annotated_data, config=configuration)
    plotting.plot_cluster_overlay(annotated_data, annotations, config=configuration)

    plot_boundaries = plots_configuration.plot_boundaries
    plot_transcripts = plots_configuration.plot_transcripts
    if plot_boundaries or plot_transcripts:
        cell_boundary_key, nucleus_boundary_key, transcripts_key = plotting.detect_spatial_keys(spatial_data)
        if plot_boundaries:
            plotting.plot_cell_nucleus_boundaries(
                spatial_data,
                cell_key=cell_boundary_key,
                nucleus_key=nucleus_boundary_key,
                config=configuration,
            )
        if plot_transcripts and transcripts_key is not None:
            # Stored in the configuration object as an immutable tuple.
            genes_to_plot = list(plots_configuration.genes_to_plot)
            if genes_to_plot:
                plotting.plot_transcripts_with_boundaries(
                    spatial_data,
                    cell_key=cell_boundary_key,
                    nucleus_key=nucleus_boundary_key,
                    transcripts_key=transcripts_key,
                    genes_to_plot=genes_to_plot,
                    config=configuration,
                )


if __name__ == "__main__":
    main()
