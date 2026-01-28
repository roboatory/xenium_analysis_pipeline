from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from spatialdata_io import xenium

from .config import Config


def load_xenium(data_directory: str | Path | None = None, config: Optional[Config] = None):
    if data_directory is None:
        if config is None:
            raise ValueError("data_directory or config must be provided.")
        data_directory = config.raw_data_dir
    data_path = Path(data_directory).resolve()
    return xenium(data_path)


def read_json_if_exists(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    with json_path.open("r") as file_handle:
        return json.load(file_handle)


def read_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing annotations file: {json_path}")
    with json_path.open("r") as file_handle:
        return json.load(file_handle)


def ensure_dir(path: str | Path) -> Path:
    directory_path = Path(path)
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path


def write_cluster_labels(
    annotated_data,
    csv_file_path: str | Path | None = None,
    config: Optional[Config] = None,
) -> Path:
    cluster_dataframe = annotated_data.obs[["leiden"]].copy()
    cluster_dataframe = cluster_dataframe.rename(columns={"leiden": "group"})

    cell_identifiers = annotated_data.obs["cell_id"].astype(str)
    cluster_dataframe.insert(0, "cell_id", cell_identifiers)
    cluster_dataframe = cluster_dataframe[["cell_id", "group"]]

    if csv_file_path is None:
        if config is None:
            raise ValueError("csv_file_path or config must be provided.")
        csv_file_path = config.results_dir / "leiden_clusters.csv"
    csv_file_path = Path(csv_file_path).expanduser()
    cluster_dataframe.to_csv(csv_file_path, sep=",", index=False)
    return csv_file_path


def write_enriched_genes(
    gene_lists_by_cluster: dict[str, list[str]],
    rows_dataframe: pd.DataFrame,
    genes_json_path: str | Path | None = None,
    genes_tsv_path: str | Path | None = None,
    config: Optional[Config] = None,
) -> tuple[Path, Path]:
    if genes_json_path is None or genes_tsv_path is None:
        if config is None:
            raise ValueError("genes_json_path/genes_tsv_path or config must be provided.")
        genes_json_path = config.results_dir / "cluster_enriched_genes.json"
        genes_tsv_path = config.results_dir / "cluster_enriched_genes.tsv"
    genes_json_path = Path(genes_json_path)
    genes_tsv_path = Path(genes_tsv_path)

    rows_dataframe.to_csv(genes_tsv_path, sep="\t", index=False)
    with genes_json_path.open("w") as file_handle:
        json.dump(gene_lists_by_cluster, file_handle, indent=2)

    return genes_json_path, genes_tsv_path


def ensure_placeholder_annotations(
    annotations_path: str | Path | None,
    clusters: list[str],
    config: Optional[Config] = None,
) -> Path:
    if annotations_path is None:
        if config is None:
            raise ValueError("annotations_path or config must be provided.")
        annotations_path = config.results_dir / "chatgpt_celltype_annotations.json"
    annotations_path = Path(annotations_path)
    if not annotations_path.exists():
        placeholder_annotations = {str(cluster): {"annotation": "", "markers_used": []} for cluster in clusters}
        with annotations_path.open("w") as file_handle:
            json.dump(placeholder_annotations, file_handle, indent=2)
    return annotations_path


def write_spatialdata_zarr(
    spatial_data,
    annotated_data,
    zarr_path: str | Path | None = None,
    config: Optional[Config] = None,
    overwrite: bool = True,
) -> Path:
    if zarr_path is None:
        if config is None:
            raise ValueError("zarr_path or config must be provided.")
        zarr_path = config.processed_data_dir / "xenium_processed_spatialdata.zarr"
    zarr_path = Path(zarr_path).expanduser()

    spatial_data["table"] = annotated_data
    if overwrite and zarr_path.exists():
        shutil.rmtree(zarr_path)

    spatial_data.write(zarr_path)
    return zarr_path
