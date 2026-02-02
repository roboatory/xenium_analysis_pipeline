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


def write_cluster_annotations(
    cluster_annotations: dict[str, dict[str, str | float]],
    configuration: Config,
) -> None:
    """Write LLM cluster annotations (label/confidence/rationale) to JSON."""

    annotations_path = configuration.results_directory / "cluster_annotations.json"
    with annotations_path.open("w") as file_handle:
        json.dump(cluster_annotations, file_handle, indent=2)


def load_enriched_genes(configuration: Config) -> dict[str, list[str]]:
    """Load enriched genes per cluster from the analysis directory JSON."""

    enriched_genes_path = (
        configuration.results_directory / "cluster_enriched_genes.json"
    )
    with enriched_genes_path.open("r") as file_handle:
        data = json.load(file_handle)

    return {
        str(cluster_id): [str(gene) for gene in genes]
        for cluster_id, genes in data.items()
    }


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
