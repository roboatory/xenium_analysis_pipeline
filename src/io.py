from __future__ import annotations

from anndata import AnnData
import json
from typing import Literal

from spatialdata import SpatialData, read_zarr
from spatialdata_io import xenium

from .config import Config


def load_xenium(configuration: Config) -> SpatialData:
    """Load Xenium spatial data using the configuration's raw data directory."""

    return xenium(configuration.raw_data_directory)


def read_spatialdata_zarr(
    configuration: Config,
    output: Literal["ingested", "processed"],
) -> SpatialData:
    """Read ingested or processed zarr file from the processed data directory."""

    return read_zarr(configuration.processed_data_directory / f"{output}.zarr")


def write_cluster_labels(
    annotated_data: AnnData,
    configuration: Config,
    cluster_key: str = "leiden",
) -> None:
    """Write the cluster labels to a CSV file."""

    cluster_dataframe = annotated_data.obs[[cluster_key]].copy()
    cluster_dataframe = cluster_dataframe.rename(columns={cluster_key: "group"})
    cluster_dataframe.insert(0, "cell_id", annotated_data.obs["cell_id"].astype(str))
    cluster_dataframe.to_csv(
        configuration.results_directory / f"{cluster_key}_clusters.csv",
        sep=",",
        index=False,
    )


def write_enriched_genes(
    gene_lists_by_cluster: dict[str, list[str]],
    configuration: Config,
) -> None:
    """Write the enriched genes to a JSON file."""

    enriched_genes_path = (
        configuration.results_directory / "cluster_enriched_genes.json"
    )

    with enriched_genes_path.open("w") as file_handle:
        json.dump(gene_lists_by_cluster, file_handle, indent=2)


def write_spatialdata_zarr(
    spatial_data: SpatialData,
    annotated_data: AnnData | None,
    configuration: Config,
    output: Literal["ingested", "processed"],
) -> None:
    """Write spatial data to ingested.zarr or processed.zarr."""

    spatial_data["table"] = annotated_data
    spatial_data.write(
        configuration.processed_data_directory / f"{output}.zarr", overwrite=True
    )
