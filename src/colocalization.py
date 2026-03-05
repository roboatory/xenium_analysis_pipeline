from __future__ import annotations

from anndata import AnnData
import numpy as np
import pandas as pd
from scipy import sparse
import squidpy as sq


def compute_observed_contact_matrices(
    annotated_data: AnnData,
    radius: float,
    label_key: str = "cell_type",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute observed contact count and row-normalized contact matrices."""

    if label_key not in annotated_data.obs:
        msg = f"Label column '{label_key}' not found in annotated_data.obs."
        raise ValueError(msg)

    labels = annotated_data.obs[label_key]
    valid_mask = labels.notna().to_numpy()
    valid_labels = labels[valid_mask].astype("category")
    categories = valid_labels.cat.categories.astype(str)

    sq.gr.spatial_neighbors(
        annotated_data,
        radius=radius,
        coord_type="generic",
        delaunay=False,
        key_added="colocalization",
        set_diag=False,
    )

    connectivities = annotated_data.obsp["colocalization_connectivities"].tocsr().copy()
    connectivities.setdiag(0)
    connectivities.eliminate_zeros()
    undirected_edges = sparse.triu(connectivities, k=1, format="coo")

    category_codes = np.full(annotated_data.n_obs, -1, dtype=np.int64)
    category_codes[valid_mask] = valid_labels.cat.codes.to_numpy()

    row_codes = category_codes[undirected_edges.row]
    col_codes = category_codes[undirected_edges.col]
    edge_mask = (row_codes >= 0) & (col_codes >= 0)
    row_codes = row_codes[edge_mask]
    col_codes = col_codes[edge_mask]

    n_categories = len(categories)
    counts = np.zeros((n_categories, n_categories), dtype=np.int64)
    np.add.at(counts, (row_codes, col_codes), 1)
    asymmetric_mask = row_codes != col_codes
    np.add.at(
        counts,
        (col_codes[asymmetric_mask], row_codes[asymmetric_mask]),
        1,
    )

    count_matrix = pd.DataFrame(counts, index=categories, columns=categories)

    row_totals = counts.sum(axis=1, keepdims=True)
    proportion_matrix = np.divide(
        counts.astype(np.float64),
        row_totals,
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_totals > 0,
    )
    row_proportion_matrix = pd.DataFrame(
        proportion_matrix,
        index=categories,
        columns=categories,
    )

    return count_matrix, row_proportion_matrix
