from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

import scanpy as sc

from src import analysis, io, plotting, preprocessing
from src.config import Config

import dask

dask.config.set({"dataframe.query-planning": True})

def main() -> None:
    configuration = Config()
    configuration.load_from_yaml(Path("config.yaml").resolve())
    configuration.create_directories()

    spatial_data = io.load_xenium(configuration)

    annotated_data = spatial_data["table"]
    annotated_data.layers["counts"] = annotated_data.X.copy()

    sc.pp.calculate_qc_metrics(annotated_data, inplace=True)

    minimum_counts = configuration.pipeline.minimum_counts
    maximum_counts_quantile = configuration.pipeline.maximum_counts_quantile
    minimum_cells = configuration.pipeline.minimum_cells

    preprocessing.filter_cells_and_genes(annotated_data, minimum_counts, maximum_counts_quantile, minimum_cells)
    preprocessing.normalize_and_scale(annotated_data, configuration.pipeline.n_top_genes)

    analysis.run_clustering(annotated_data, configuration.pipeline.n_components, configuration.pipeline.leiden_resolution)
    analysis.run_umap(annotated_data)
    analysis.rank_genes(annotated_data)

    clusters = pd.unique(annotated_data.obs["leiden"])
    enriched_gene_lists = analysis.compute_enriched_genes(
        annotated_data,
        clusters=clusters,
        top_n=configuration.pipeline.rank_top_n,
        minimum_logarithm_fold_change=configuration.pipeline.minimum_logarithm_fold_change,
        maximum_adjusted_p_value=configuration.pipeline.maximum_adjusted_p_value,
    )

    io.write_cluster_labels(annotated_data, configuration)
    io.write_enriched_genes(enriched_gene_lists, configuration)
    io.write_spatialdata_zarr(spatial_data, annotated_data, configuration)

    plotting.plot_qc_hist(
        annotated_data,
        cutoffs=[minimum_counts, int(np.quantile(annotated_data.obs["total_counts"], maximum_counts_quantile))],
        cutoff_colors=["crimson", "goldenrod"],
        config=configuration,
    )
    plotting.plot_umap_leiden(annotated_data, config=configuration)
    plotting.plot_rank_genes_dotplot(annotated_data, config=configuration)
    plotting.plot_cluster_overlay(annotated_data, annotations, config=configuration)

    plot_boundaries = configuration.plots.plot_boundaries
    plot_transcripts = configuration.plots.plot_transcripts
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
            genes_to_plot = list(configuration.plots.genes_to_plot)
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
