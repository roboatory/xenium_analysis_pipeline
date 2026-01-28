from __future__ import annotations

import numpy as np
import scanpy as sc


def configure_scanpy(verbosity: int = 2, dpi: int = 90, facecolor: str = "white") -> None:
    sc.settings.verbosity = verbosity
    sc.settings.set_figure_params(dpi=dpi, facecolor=facecolor)


def init_adata(spatial_data):
    annotated_data = spatial_data["table"]
    annotated_data.layers["counts"] = annotated_data.X.copy()
    return annotated_data


def filter_cells_and_genes(
    annotated_data,
    min_counts: int = 200,
    max_counts_quantile: float = 0.99,
    min_cells: int = 100,
) -> float:
    total_count_threshold = np.quantile(annotated_data.obs["total_counts"], max_counts_quantile)
    sc.pp.filter_cells(annotated_data, min_counts=min_counts)
    sc.pp.filter_cells(annotated_data, max_counts=total_count_threshold)
    sc.pp.filter_genes(annotated_data, min_cells=min_cells)
    return total_count_threshold


def normalize_and_scale(
    annotated_data,
    n_top_genes: int = 2000,
    hvg_flavor: str = "seurat_v3",
    zero_center: bool = False,
    max_value: float = 10,
) -> None:
    sc.pp.highly_variable_genes(annotated_data, flavor=hvg_flavor, n_top_genes=n_top_genes)
    sc.pp.normalize_total(annotated_data)
    sc.pp.log1p(annotated_data)
    annotated_data.layers["log_norm"] = annotated_data.X.copy()
    sc.pp.scale(annotated_data, zero_center=zero_center, max_value=max_value)
