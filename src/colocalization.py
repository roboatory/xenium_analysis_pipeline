from __future__ import annotations

from anndata import AnnData
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import squidpy as sq


def compute_neighborhood_composition(
    annotated_data: AnnData,
    cluster_key: str,
    radius: float,
    spatial_key: str = "spatial",
    composition_key: str = "neighborhood_composition",
    composition_labels_key: str = "neighborhood_composition_labels",
) -> pd.DataFrame:
    """Compute neighborhood composition vectors from a Squidpy spatial graph."""

    if radius <= 0:
        raise ValueError(f"Radius must be positive. Found `{radius}`.")
    if cluster_key not in annotated_data.obs:
        raise KeyError(f"Missing cluster key `{cluster_key}` in `annotated_data.obs`.")
    if spatial_key not in annotated_data.obsm:
        raise KeyError(
            f"Missing spatial coordinates key `{spatial_key}` in `annotated_data.obsm`."
        )

    cluster_values = annotated_data.obs[cluster_key].astype("category")
    cluster_categories = cluster_values.cat.categories.astype(str)
    cluster_codes = cluster_values.cat.codes.to_numpy()

    sq.gr.spatial_neighbors(
        annotated_data,
        spatial_key=spatial_key,
        coord_type="generic",
        radius=radius,
    )
    if "spatial_connectivities" not in annotated_data.obsp:
        raise KeyError(
            "Missing `spatial_connectivities` in `annotated_data.obsp` after "
            "`sq.gr.spatial_neighbors`."
        )

    connectivities = annotated_data.obsp["spatial_connectivities"].tocsr(copy=True)
    connectivities.setdiag(0)
    connectivities.eliminate_zeros()

    n_cells = annotated_data.n_obs
    n_cell_types = len(cluster_categories)
    one_hot = np.zeros((n_cells, n_cell_types), dtype=np.float32)
    valid_mask = cluster_codes >= 0
    one_hot[np.flatnonzero(valid_mask), cluster_codes[valid_mask]] = 1.0

    neighbor_counts = np.asarray(connectivities @ one_hot, dtype=np.float32)
    totals = neighbor_counts.sum(axis=1, dtype=np.float32)
    composition = np.zeros_like(neighbor_counts, dtype=np.float32)

    has_neighbors = totals > 0
    composition[has_neighbors] = (
        neighbor_counts[has_neighbors] / totals[has_neighbors, None]
    )

    isolated = (~has_neighbors) & valid_mask
    composition[np.flatnonzero(isolated), cluster_codes[isolated]] = 1.0

    annotated_data.obsm[composition_key] = composition
    annotated_data.uns[composition_labels_key] = cluster_categories.tolist()

    return pd.DataFrame(
        composition,
        index=annotated_data.obs_names,
        columns=cluster_categories,
    )


def assign_spatial_domains(
    annotated_data: AnnData,
    n_clusters: int = 10,
    domain_key: str = "spatial_domain",
    composition_key: str = "neighborhood_composition",
) -> pd.Series:
    """Cluster neighborhood composition vectors into spatial domain labels."""

    if composition_key not in annotated_data.obsm:
        raise KeyError(
            f"Missing composition key `{composition_key}` in `annotated_data.obsm`."
        )
    if n_clusters < 2:
        raise ValueError("`n_clusters` must be >= 2 for spatial domain clustering.")

    composition_matrix = np.asarray(annotated_data.obsm[composition_key])
    if composition_matrix.ndim != 2:
        raise ValueError("Neighborhood composition matrix must be 2-dimensional.")

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init="auto",
        random_state=0,
    )
    labels = kmeans.fit_predict(composition_matrix).astype(str)

    annotated_data.obs[domain_key] = pd.Categorical(labels)
    return annotated_data.obs[domain_key]
