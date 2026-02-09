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
) -> None:
    """Compute per-cell neighborhood cell-type proportions within a spatial radius."""

    if (
        cluster_key == "cell_type"
        and cluster_key not in annotated_data.obs
        and "leiden" in annotated_data.obs
    ):
        cluster_key = "leiden"

    cluster_values = annotated_data.obs[cluster_key].astype("category")
    cluster_categories = cluster_values.cat.categories.astype(str)
    cluster_codes = cluster_values.cat.codes.to_numpy()
    valid_mask = cluster_codes >= 0

    sq.gr.spatial_neighbors(
        annotated_data,
        spatial_key=spatial_key,
        coord_type="generic",
        radius=radius,
        delaunay=True,
    )

    connectivities = annotated_data.obsp["spatial_connectivities"].tocsr(copy=True)
    connectivities.setdiag(0)
    connectivities.eliminate_zeros()

    one_hot = np.zeros(
        (annotated_data.n_obs, len(cluster_categories)), dtype=np.float32
    )
    valid_rows = np.flatnonzero(valid_mask)
    one_hot[valid_rows, cluster_codes[valid_rows]] = 1.0

    neighbor_counts = np.asarray(connectivities @ one_hot, dtype=np.float32)
    totals = neighbor_counts.sum(axis=1, keepdims=True, dtype=np.float32)
    composition = np.zeros_like(neighbor_counts, dtype=np.float32)
    np.divide(neighbor_counts, totals, out=composition, where=totals > 0)

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
