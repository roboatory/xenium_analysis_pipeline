from __future__ import annotations

# Dask config must run before any import that pulls in dask.
import dask.config

dask.config.set({"dataframe.query-planning": True})

from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from src import analysis, annotation, io, plotting, preprocessing  # noqa: E402
from src.config import Config  # noqa: E402


def main() -> None:
    configuration = Config()
    configuration.load_from_yaml(Path("config.yaml").resolve())
    configuration.create_directories()

    processed_path = configuration.processed_data_directory / "processed.zarr"
    if not processed_path.exists():
        ingested_path = configuration.processed_data_directory / "ingested.zarr"
        if not ingested_path.exists():
            spatial_data = io.load_xenium(configuration)
            annotated_data = spatial_data["table"]
            del spatial_data["morphology_focus"]
            io.write_spatialdata_zarr(
                spatial_data, annotated_data, configuration, "ingested"
            )

        spatial_data = io.read_spatialdata_zarr(configuration, "ingested")
        annotated_data = spatial_data["table"]

        if configuration.plots.plot_boundaries:
            plotting.plot_cell_and_nucleus_boundaries(spatial_data, configuration)
        if configuration.plots.plot_transcripts:
            plotting.plot_transcripts(
                spatial_data,
                configuration,
                list[str](configuration.plots.genes_to_plot),
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
        io.write_spatialdata_zarr(
            spatial_data, annotated_data, configuration, "processed"
        )

    spatial_data = io.read_spatialdata_zarr(configuration, "processed")
    annotated_data = spatial_data["table"]

    enriched_gene_lists = io.load_enriched_genes(configuration)
    cluster_annotations = annotation.annotate_clusters_with_llm(enriched_gene_lists)
    io.write_cluster_annotations(cluster_annotations, configuration)

    annotated_data.obs["cell_type"] = (
        annotated_data.obs["leiden"]
        .astype(str)
        .map({c: v["cell_type"] for c, v in cluster_annotations.items()})
    )

    spatial_data["table"] = annotated_data


if __name__ == "__main__":
    main()
