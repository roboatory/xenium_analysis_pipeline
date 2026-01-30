from __future__ import annotations

import numpy as np
import scanpy as sc
from anndata import AnnData


def filter_cells_and_genes(
    annotated_data: AnnData,
    minimum_counts: int,
    maximum_counts_quantile: float,
    minimum_cells: int,
) -> None:
    """Filter cells and genes based on the annotated data."""

    maximum_counts = int(
        np.quantile(annotated_data.obs["total_counts"], maximum_counts_quantile)
    )

    sc.pp.filter_cells(annotated_data, min_counts=minimum_counts)
    sc.pp.filter_cells(annotated_data, max_counts=maximum_counts)
    sc.pp.filter_genes(annotated_data, min_cells=minimum_cells)


def normalize_and_scale(
    annotated_data: AnnData,
    n_top_genes: int,
    algorithm: str = "seurat_v3",
    zero_center: bool = False,
    max_value: float = 10,
) -> None:
    """Normalize and scale the annotated data."""

    sc.pp.highly_variable_genes(
        annotated_data, flavor=algorithm, n_top_genes=n_top_genes
    )
    sc.pp.normalize_total(annotated_data)
    sc.pp.log1p(annotated_data)
    annotated_data.layers["log_normalized"] = annotated_data.X.copy()
    sc.pp.scale(annotated_data, zero_center=zero_center, max_value=max_value)
