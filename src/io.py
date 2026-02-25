from __future__ import annotations

from anndata import AnnData, concat
import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from spatialdata import SpatialData, read_zarr
from spatialdata_io import xenium

from .config import Config


def load_xenium(raw_data_directory: Path) -> SpatialData:
    """Load Xenium spatial data from one raw directory."""

    return xenium(raw_data_directory)


def read_sample_spatialdata_zarr(configuration: Config, sample_id: str) -> SpatialData:
    """Read one cached sample SpatialData zarr."""

    sample_path = (
        configuration.processed_data_directory / "samples" / f"{sample_id}.zarr"
    )
    return read_zarr(sample_path)


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


def write_cluster_labels(
    annotated_data: AnnData,
    configuration: Config,
    cluster_key: str = "leiden",
) -> None:
    """Write the cluster labels to a CSV file."""

    cluster_dataframe = annotated_data.obs[[cluster_key]].copy()
    cluster_dataframe = cluster_dataframe.rename(columns={cluster_key: "group"})
    if "cell_id" in annotated_data.obs.columns:
        cell_ids = annotated_data.obs["cell_id"].astype(str)
    else:
        cell_ids = annotated_data.obs_names.astype(str)
    cluster_dataframe.insert(0, "cell_id", cell_ids)
    if "sample_id" in annotated_data.obs.columns:
        cluster_dataframe.insert(
            1, "sample_id", annotated_data.obs["sample_id"].astype(str)
        )
    cluster_dataframe.to_csv(
        configuration.results_directory / f"{cluster_key}_clusters.csv",
        sep=",",
        index=False,
    )


def _write_json(path, payload: Any) -> None:
    """Write JSON payload to disk."""

    with path.open("w") as file_handle:
        json.dump(payload, file_handle, indent=2)


def _build_obs_label_dataframe(
    annotated_data: AnnData,
    value_key: str,
    output_column: str,
):
    """Build a labels dataframe with canonical `cell_id` first column."""

    labels_dataframe = annotated_data.obs[[value_key]].copy()
    labels_dataframe = labels_dataframe.rename(columns={value_key: output_column})
    if "cell_id" in annotated_data.obs.columns:
        cell_ids = annotated_data.obs["cell_id"].astype(str)
    else:
        cell_ids = annotated_data.obs_names.astype(str)
    labels_dataframe.insert(0, "cell_id", cell_ids)
    if "sample_id" in annotated_data.obs.columns:
        labels_dataframe.insert(
            1, "sample_id", annotated_data.obs["sample_id"].astype(str)
        )
    return labels_dataframe


def write_annotations(
    annotations: dict[str, dict[str, str | float]],
    configuration: Config,
    target: str,
    domain_key: str = "spatial_domain",
) -> None:
    """Write cluster/domain annotation JSON artifacts."""

    annotation_paths = {
        "cluster": configuration.results_directory / "cluster_annotations.json",
        "domain": configuration.results_directory / f"{domain_key}_annotations.json",
    }
    annotations_path = annotation_paths[target]
    _write_json(annotations_path, annotations)


def write_sample_spatialdata_zarr(
    spatial_data: SpatialData,
    configuration: Config,
    sample_id: str,
) -> None:
    """Write one sample SpatialData cache zarr under processed/samples."""

    sample_directory = configuration.processed_data_directory / "samples"
    sample_directory.mkdir(parents=True, exist_ok=True)
    target_path = sample_directory / f"{sample_id}.zarr"
    temporary_path = target_path.parent / f".{target_path.name}.tmp-{uuid4().hex}"
    spatial_data.write(temporary_path, overwrite=True)
    try:
        if target_path.exists():
            shutil.rmtree(target_path)
        temporary_path.rename(target_path)
    finally:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)


def pool_tables(sample_tables: list[AnnData]) -> AnnData:
    """Concatenate filtered sample tables into a single pooled AnnData."""

    return concat(
        sample_tables,
        axis=0,
        join="outer",
        merge="same",
        uns_merge="first",
        fill_value=0,
        index_unique="::",
    )


def write_analysis_artifact(
    configuration: Config,
    target: str,
    enriched_genes: dict[str, list[str]] | None = None,
    annotated_data: AnnData | None = None,
    domain_key: str = "spatial_domain",
) -> None:
    """Write enriched genes JSON or spatial domain labels CSV."""

    if target == "enriched_genes":
        _write_json(
            configuration.results_directory / "cluster_enriched_genes.json",
            enriched_genes,
        )
        return

    if target == "spatial_domains":
        domain_dataframe = _build_obs_label_dataframe(
            annotated_data,
            value_key=domain_key,
            output_column="domain",
        )
        domain_dataframe.to_csv(
            configuration.results_directory / f"{domain_key}_labels.csv",
            index=False,
        )
        return
