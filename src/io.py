from __future__ import annotations

from anndata import AnnData
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from spatialdata import SpatialData
from spatialdata_io import xenium

from .config import Config


def load_xenium(configuration: Config) -> SpatialData:
    """Load Xenium spatial data using the configuration's raw data directory."""

    return xenium(configuration.raw_data_directory)


def write_cluster_labels(annotated_data: AnnData, configuration: Config) -> None:
    """Write the cluster labels to a CSV file."""
    
    cluster_dataframe = annotated_data.obs[["leiden"]].copy()
    cluster_dataframe = cluster_dataframe.rename(columns={"leiden": "group"})
    cluster_dataframe.insert(0, "cell_id", annotated_data.obs["cell_id"].astype(str))
    cluster_dataframe.to_csv(configuration.results_directory / "leiden_clusters.csv", sep=",", index=False)


def write_enriched_genes(gene_lists_by_cluster: dict[str, list[str]], configuration: Config,) -> None:
    """Write the enriched genes to a JSON file."""

    enriched_genes_path = configuration.results_directory / "cluster_enriched_genes.json"

    with enriched_genes_path.open("w") as file_handle:
        json.dump(gene_lists_by_cluster, file_handle, indent=2)


def write_spatialdata_zarr(spatial_data: SpatialData, annotated_data: AnnData, configuration: Config) -> None:
    """Write the spatial data to a Zarr file."""

    zarr_path = configuration.processed_data_directory / "processed_data.zarr"

    spatial_data["table"] = annotated_data
    spatial_data.write(zarr_path)
