from __future__ import annotations

import numpy as np
import scanpy as sc
from anndata import AnnData

from .logging_utils import get_logger

logger = get_logger(__name__)


def filter_cells_and_genes(
    annotated_data: AnnData,
    minimum_counts: int,
    maximum_counts_quantile: float,
    minimum_cells: int,
) -> None:
    """Filter cells and genes based on the annotated data."""

    logger.debug(
        "filtering cells and genes with minimum_counts=%s, maximum_counts_quantile=%s, minimum_cells=%s",
        minimum_counts,
        maximum_counts_quantile,
        minimum_cells,
    )
    sc.pp.filter_cells(annotated_data, min_counts=minimum_counts)
    sc.pp.filter_cells(
        annotated_data,
        max_counts=int(
            np.quantile(annotated_data.obs["total_counts"], maximum_counts_quantile)
        ),
    )
    sc.pp.filter_genes(annotated_data, min_cells=minimum_cells)
    logger.info(
        "QC retained %s cells and %s genes",
        annotated_data.n_obs,
        annotated_data.n_vars,
    )


def normalize_and_scale(
    annotated_data: AnnData,
) -> None:
    """Normalize and scale the annotated data."""

    logger.debug("normalizing and scaling %s cells", annotated_data.n_obs)
    sc.pp.highly_variable_genes(
        annotated_data,
        flavor="seurat_v3",
        n_top_genes=2000,
    )
    sc.pp.normalize_total(annotated_data)
    sc.pp.log1p(annotated_data)
    annotated_data.layers["log_normalized"] = annotated_data.X.copy()
    sc.pp.scale(annotated_data, zero_center=False, max_value=10)
    logger.debug("normalization and scaling complete")
