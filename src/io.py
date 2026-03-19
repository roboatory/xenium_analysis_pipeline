from __future__ import annotations

from anndata import AnnData
import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from spatialdata import SpatialData, read_zarr

from .config import Configuration


def read_spatialdata_zarr(
    configuration: Configuration,
) -> SpatialData:
    """Read processed zarr file from the processed data directory."""

    return read_zarr(configuration.processed_data_directory / "processed.zarr")


def write_spatialdata_zarr(
    configuration: Configuration,
    spatial_data: SpatialData,
) -> None:
    """Write processed spatial zarr using a simple temp-then-replace flow."""

    target_path = configuration.processed_data_directory / "processed.zarr"
    temporary_path = target_path.parent / f".{target_path.name}.tmp-{uuid4().hex}"

    spatial_data.write(temporary_path, overwrite=True)

    try:
        if target_path.exists():
            shutil.rmtree(target_path)
        temporary_path.rename(target_path)
    finally:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)


def read_enriched_genes(
    configuration: Configuration,
) -> dict[str, list[str]]:
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


def write_enriched_genes(
    configuration: Configuration,
    enriched_genes: dict[str, list[str]],
) -> None:
    """Write enriched genes JSON artifact."""

    enriched_genes_path = (
        configuration.results_directory / "cluster_enriched_genes.json"
    )
    with enriched_genes_path.open("w") as file_handle:
        json.dump(enriched_genes, file_handle, indent=2)


def write_annotations(
    configuration: Configuration,
    annotations: dict[str, dict[str, str | float]],
    target: str,
) -> None:
    """Write cluster/domain annotation JSON artifacts."""

    annotation_paths = {
        "cluster": configuration.results_directory / "cluster_annotations.json",
        "domain": configuration.results_directory / "spatial_domain_annotations.json",
    }
    annotations_path = annotation_paths[target]
    with annotations_path.open("w") as file_handle:
        json.dump(annotations, file_handle, indent=2)


def write_labels(
    configuration: Configuration,
    annotated_data: AnnData,
    target: str,
) -> None:
    """Write cluster/domain label CSV artifacts."""

    if target == "cluster":
        label_column = "leiden"
        output_path = configuration.results_directory / "leiden_clusters.csv"
    elif target == "domain":
        label_column = "spatial_domain"
        output_path = configuration.results_directory / "spatial_domain_labels.csv"

    dataframe = annotated_data.obs[[label_column]].rename(
        columns={label_column: "group"}
    )
    dataframe.insert(0, "cell_id", annotated_data.obs["cell_id"])
    dataframe.to_csv(output_path, index=False)


def save_state(path: Path, state_payload: dict[str, Any]) -> None:
    """Write run configuration snapshot JSON."""

    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(state_payload, file_handle, indent=2, sort_keys=True)
