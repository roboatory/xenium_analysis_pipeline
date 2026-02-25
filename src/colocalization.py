from __future__ import annotations

from anndata import AnnData
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import squidpy as sq


def compute_neighborhood_composition(
    annotated_data: AnnData,
    radius: float,
    cluster_key: str = "cell_type",
    spatial_key: str = "spatial",
    composition_key: str = "neighborhood_composition",
    sample_key: str = "sample_id",
) -> None:
    """Compute per-cell neighborhood cell-type proportions within a spatial radius."""

    cluster_values = annotated_data.obs[cluster_key].astype("category")
    cluster_categories = cluster_values.cat.categories.astype(str)
    cluster_codes = cluster_values.cat.codes.to_numpy()
    valid_mask = cluster_codes >= 0

    one_hot = np.zeros(
        (annotated_data.n_obs, len(cluster_categories)),
        dtype=np.float32,
    )
    valid_rows = np.flatnonzero(valid_mask)
    one_hot[valid_rows, cluster_codes[valid_rows]] = 1.0

    composition = np.zeros_like(one_hot, dtype=np.float32)
    sample_values = annotated_data.obs[sample_key].astype(str).to_numpy()
    for sample_id in sorted(pd.unique(sample_values)):
        sample_indices = np.flatnonzero(sample_values == sample_id)
        if sample_indices.size <= 1:
            continue

        sample_data = annotated_data[sample_indices].copy()
        sq.gr.spatial_neighbors(
            sample_data,
            spatial_key=spatial_key,
            coord_type="generic",
            radius=radius,
            delaunay=True,
        )

        connectivities = sample_data.obsp["spatial_connectivities"].tocsr(copy=True)
        connectivities.setdiag(0)
        connectivities.eliminate_zeros()

        neighbor_counts = np.asarray(
            connectivities @ one_hot[sample_indices],
            dtype=np.float32,
        )
        totals = neighbor_counts.sum(axis=1, keepdims=True, dtype=np.float32)
        np.divide(
            neighbor_counts,
            totals,
            out=composition[sample_indices],
            where=totals > 0,
        )

    annotated_data.obsm[composition_key] = composition


def assign_spatial_domains(
    annotated_data: AnnData,
    n_clusters: int = 10,
    domain_key: str = "spatial_domain",
    composition_key: str = "neighborhood_composition",
) -> None:
    """Cluster neighborhood composition vectors into spatial domain labels."""

    composition_matrix = np.asarray(annotated_data.obsm[composition_key])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(composition_matrix).astype(str)

    annotated_data.obs[domain_key] = pd.Categorical(labels)
