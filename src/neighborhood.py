from __future__ import annotations

from anndata import AnnData
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import squidpy as sq


def compute_neighborhood_composition(
    annotated_data: AnnData,
    radius: float,
) -> None:
    """Compute per-cell neighborhood cell-type proportions within a spatial radius."""

    cell_types = annotated_data.obs["cell_type"].astype("category")
    categories = cell_types.cat.categories.astype(str)
    codes = cell_types.cat.codes.to_numpy()

    sq.gr.spatial_neighbors(
        annotated_data,
        radius=radius,
        coord_type="generic",
        delaunay=True,
    )

    connectivities = annotated_data.obsp["spatial_connectivities"].tocsr().copy()
    connectivities.setdiag(0)
    connectivities.eliminate_zeros()

    one_hot = np.eye(len(categories), dtype=np.float32)[codes]
    one_hot[codes < 0] = 0

    neighbor_counts = connectivities @ one_hot
    totals = neighbor_counts.sum(axis=1, keepdims=True)
    composition = np.divide(
        neighbor_counts,
        totals,
        out=np.zeros_like(neighbor_counts),
        where=totals > 0,
    )

    annotated_data.obsm["neighborhood_composition"] = composition


def assign_spatial_domains(
    annotated_data: AnnData,
    n_clusters: int = 10,
) -> None:
    """Cluster neighborhood composition vectors into spatial domain labels."""

    composition_matrix = np.asarray(annotated_data.obsm["neighborhood_composition"])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(composition_matrix).astype(str)

    annotated_data.obs["spatial_domain"] = pd.Categorical(labels)
