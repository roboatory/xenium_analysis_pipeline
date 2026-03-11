from __future__ import annotations

from anndata import AnnData
import json
import pandas as pd
import shutil
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


def write_colocalization_matrices(
    configuration: Configuration,
    counts: pd.DataFrame,
    proportions: pd.DataFrame,
) -> None:
    """Write cell-type contact matrix artifacts."""

    counts_path = configuration.results_directory / "cell_type_contact_counts.csv"
    proportions_path = (
        configuration.results_directory / "cell_type_contact_row_proportions.csv"
    )
    counts.to_csv(counts_path)
    proportions.to_csv(proportions_path)


def write_colocalization_permutation_matrices(
    configuration: Configuration,
    expected_counts: pd.DataFrame,
    fold_enrichment: pd.DataFrame,
    log2_fold_enrichment: pd.DataFrame,
    empirical_p_values: pd.DataFrame,
    fdr: pd.DataFrame,
    significant_mask: pd.DataFrame,
) -> None:
    """Write colocalization permutation and significance matrices."""

    matrix_paths = {
        "expected_counts": configuration.results_directory
        / "cell_type_contact_expected_counts.csv",
        "fold_enrichment": configuration.results_directory
        / "cell_type_contact_fold_enrichment.csv",
        "log2_fold_enrichment": configuration.results_directory
        / "cell_type_contact_log2_fold_enrichment.csv",
        "empirical_p_values": configuration.results_directory
        / "cell_type_contact_empirical_p_values.csv",
        "fdr": configuration.results_directory / "cell_type_contact_fdr.csv",
        "significant_mask": configuration.results_directory
        / "cell_type_contact_significant_mask.csv",
    }
    expected_counts.to_csv(matrix_paths["expected_counts"])
    fold_enrichment.to_csv(matrix_paths["fold_enrichment"])
    log2_fold_enrichment.to_csv(matrix_paths["log2_fold_enrichment"])
    empirical_p_values.to_csv(matrix_paths["empirical_p_values"])
    fdr.to_csv(matrix_paths["fdr"])
    significant_mask.astype(int).to_csv(matrix_paths["significant_mask"])


def write_colocalization_significance_tables(
    configuration: Configuration,
    pair_statistics_all: pd.DataFrame,
    pair_statistics_significant: pd.DataFrame,
    excluded_low_count_types: pd.DataFrame,
) -> None:
    """Write colocalization pair-level significance tables."""

    pair_statistics_all.to_csv(
        configuration.results_directory / "cell_type_contact_pair_statistics_all.csv",
        index=False,
    )
    pair_statistics_significant.to_csv(
        configuration.results_directory
        / "cell_type_contact_pair_statistics_significant.csv",
        index=False,
    )
    excluded_low_count_types.to_csv(
        configuration.results_directory
        / "cell_type_contact_excluded_low_count_types.csv",
        index=False,
    )
