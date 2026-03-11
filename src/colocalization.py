from __future__ import annotations

from anndata import AnnData
import numpy as np
import pandas as pd
from scipy import sparse
import squidpy as sq

FDR_ALPHA = 0.05
RANDOM_SEED = 0


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

    undirected_edges = _build_undirected_edges(annotated_data, radius)

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


def compute_permutation_significance(
    annotated_data: AnnData,
    radius: float,
    label_key: str = "cell_type",
    number_of_permutations: int = 1000,
    minimum_cells: int = 30,
) -> dict[str, pd.DataFrame]:
    """Compute permutation-based significance for cell-type contact enrichment."""

    if label_key not in annotated_data.obs:
        msg = f"Label column '{label_key}' not found in annotated_data.obs."
        raise ValueError(msg)
    if number_of_permutations <= 0:
        msg = "number_of_permutations must be positive."
        raise ValueError(msg)

    labels = annotated_data.obs[label_key]
    valid_labels = labels[labels.notna()].astype(str)
    cell_counts = valid_labels.value_counts().sort_index()
    tested_types = cell_counts[cell_counts >= minimum_cells].index.astype(str).tolist()
    excluded_types = cell_counts[cell_counts < minimum_cells].sort_index()

    excluded_types_dataframe = pd.DataFrame(
        {
            "cell_type": excluded_types.index.astype(str),
            "cell_count": excluded_types.to_numpy(dtype=np.int64),
        }
    )

    if len(tested_types) < 2:
        empty_matrix = pd.DataFrame(
            index=tested_types, columns=tested_types, dtype=float
        )
        empty_pairs = _empty_pair_statistics_dataframe()
        return {
            "expected_counts": empty_matrix.copy(),
            "fold_enrichment": empty_matrix.copy(),
            "log2_fold_enrichment": empty_matrix.copy(),
            "empirical_p_values": empty_matrix.copy(),
            "fdr": empty_matrix.copy(),
            "significant_mask": empty_matrix.astype(bool),
            "pair_statistics_all": empty_pairs.copy(),
            "pair_statistics_significant": empty_pairs.copy(),
            "excluded_low_count_types": excluded_types_dataframe,
        }

    undirected_edges = _build_undirected_edges(annotated_data, radius)
    edge_rows = undirected_edges.row.astype(np.int64)
    edge_cols = undirected_edges.col.astype(np.int64)

    type_to_code = {cell_type: index for index, cell_type in enumerate(tested_types)}
    full_codes = (
        labels.astype("string").map(type_to_code).fillna(-1).to_numpy(dtype=np.int64)
    )

    eligible_cell_mask = full_codes >= 0
    eligible_cell_indices = np.flatnonzero(eligible_cell_mask)
    eligible_codes = full_codes[eligible_cell_indices]

    edge_has_eligible_endpoints = (
        eligible_cell_mask[edge_rows] & eligible_cell_mask[edge_cols]
    )
    tested_edge_rows = edge_rows[edge_has_eligible_endpoints]
    tested_edge_cols = edge_cols[edge_has_eligible_endpoints]

    n_types = len(tested_types)
    if tested_edge_rows.size == 0:
        zeros = np.zeros((n_types, n_types), dtype=np.float64)
        matrix = pd.DataFrame(zeros, index=tested_types, columns=tested_types)
        nan_matrix = matrix.copy()
        nan_matrix.loc[:, :] = np.nan
        empty_pairs = _empty_pair_statistics_dataframe()
        return {
            "expected_counts": matrix.copy(),
            "fold_enrichment": nan_matrix.copy(),
            "log2_fold_enrichment": nan_matrix.copy(),
            "empirical_p_values": nan_matrix.copy(),
            "fdr": nan_matrix.copy(),
            "significant_mask": matrix.astype(bool),
            "pair_statistics_all": empty_pairs.copy(),
            "pair_statistics_significant": empty_pairs.copy(),
            "excluded_low_count_types": excluded_types_dataframe,
        }

    eligible_position = np.full(annotated_data.n_obs, -1, dtype=np.int64)
    eligible_position[eligible_cell_indices] = np.arange(
        eligible_cell_indices.size, dtype=np.int64
    )
    edge_row_positions = eligible_position[tested_edge_rows]
    edge_col_positions = eligible_position[tested_edge_cols]

    observed_row_codes = eligible_codes[edge_row_positions]
    observed_col_codes = eligible_codes[edge_col_positions]
    observed_counts = _build_symmetric_contact_counts(
        observed_row_codes,
        observed_col_codes,
        n_types,
    )

    expected_sum = np.zeros((n_types, n_types), dtype=np.float64)
    exceed_counts = np.zeros((n_types, n_types), dtype=np.int64)
    random_generator = np.random.default_rng(RANDOM_SEED)

    for _ in range(number_of_permutations):
        permuted_codes = random_generator.permutation(eligible_codes)
        permutation_row_codes = permuted_codes[edge_row_positions]
        permutation_col_codes = permuted_codes[edge_col_positions]
        permutation_counts = _build_symmetric_contact_counts(
            permutation_row_codes,
            permutation_col_codes,
            n_types,
        )
        expected_sum += permutation_counts
        exceed_counts += permutation_counts >= observed_counts

    expected_counts = expected_sum / float(number_of_permutations)
    empirical_p_values = (exceed_counts.astype(np.float64) + 1.0) / float(
        number_of_permutations + 1
    )

    fold_enrichment = np.divide(
        observed_counts.astype(np.float64),
        expected_counts,
        out=np.full_like(expected_counts, np.nan, dtype=np.float64),
        where=expected_counts > 0,
    )
    log2_fold_enrichment = np.where(
        fold_enrichment > 0,
        np.log2(fold_enrichment),
        np.nan,
    )

    fdr_matrix = np.full((n_types, n_types), np.nan, dtype=np.float64)
    significant_mask = np.zeros((n_types, n_types), dtype=bool)

    upper_triangle_indices = np.triu_indices(n_types, k=1)
    upper_triangle_p_values = empirical_p_values[upper_triangle_indices]
    upper_triangle_fdr = _benjamini_hochberg(upper_triangle_p_values)
    fdr_matrix[upper_triangle_indices] = upper_triangle_fdr
    fdr_matrix[(upper_triangle_indices[1], upper_triangle_indices[0])] = (
        upper_triangle_fdr
    )
    np.fill_diagonal(fdr_matrix, np.nan)

    upper_triangle_fold_enrichment = fold_enrichment[upper_triangle_indices]
    upper_triangle_significant = (
        np.isfinite(upper_triangle_fdr)
        & (upper_triangle_fdr <= FDR_ALPHA)
        & np.isfinite(upper_triangle_fold_enrichment)
        & (upper_triangle_fold_enrichment > 1.0)
    )
    significant_mask[upper_triangle_indices] = upper_triangle_significant
    significant_mask[(upper_triangle_indices[1], upper_triangle_indices[0])] = (
        upper_triangle_significant
    )
    np.fill_diagonal(significant_mask, False)

    tested_counts = cell_counts.loc[tested_types].to_dict()
    pair_statistics_all = _build_pair_statistics_dataframe(
        tested_types,
        observed_counts,
        expected_counts,
        fold_enrichment,
        log2_fold_enrichment,
        empirical_p_values,
        fdr_matrix,
        significant_mask,
        tested_counts,
        number_of_permutations,
    )
    pair_statistics_significant = pair_statistics_all[
        pair_statistics_all["is_significant"]
    ].copy()

    index = pd.Index(tested_types, name="cell_type")
    return {
        "expected_counts": pd.DataFrame(expected_counts, index=index, columns=index),
        "fold_enrichment": pd.DataFrame(fold_enrichment, index=index, columns=index),
        "log2_fold_enrichment": pd.DataFrame(
            log2_fold_enrichment, index=index, columns=index
        ),
        "empirical_p_values": pd.DataFrame(
            empirical_p_values, index=index, columns=index
        ),
        "fdr": pd.DataFrame(fdr_matrix, index=index, columns=index),
        "significant_mask": pd.DataFrame(significant_mask, index=index, columns=index),
        "pair_statistics_all": pair_statistics_all,
        "pair_statistics_significant": pair_statistics_significant,
        "excluded_low_count_types": excluded_types_dataframe,
    }


def _build_undirected_edges(
    annotated_data: AnnData, radius: float
) -> sparse.coo_matrix:
    """Build undirected edges from a radius-based neighbor graph."""

    sq.gr.spatial_neighbors(
        annotated_data,
        radius=radius,
        coord_type="generic",
        delaunay=True,
    )
    connectivities = annotated_data.obsp["spatial_connectivities"].tocsr().copy()
    connectivities.setdiag(0)
    connectivities.eliminate_zeros()
    return sparse.triu(connectivities, k=1, format="coo")


def _build_symmetric_contact_counts(
    row_codes: np.ndarray,
    col_codes: np.ndarray,
    n_types: int,
) -> np.ndarray:
    """Build symmetric contact count matrix from undirected edge type pairs."""

    counts = np.zeros((n_types, n_types), dtype=np.int64)
    np.add.at(counts, (row_codes, col_codes), 1)
    off_diagonal = row_codes != col_codes
    np.add.at(counts, (col_codes[off_diagonal], row_codes[off_diagonal]), 1)
    return counts


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg correction to a 1D p-value array."""

    n_tests = p_values.size
    if n_tests == 0:
        return np.array([], dtype=np.float64)

    order = np.argsort(p_values)
    sorted_p_values = p_values[order]
    ranks = np.arange(1, n_tests + 1, dtype=np.float64)
    adjusted = sorted_p_values * n_tests / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    corrected = np.empty(n_tests, dtype=np.float64)
    corrected[order] = adjusted
    return corrected


def _build_pair_statistics_dataframe(
    tested_types: list[str],
    observed_counts: np.ndarray,
    expected_counts: np.ndarray,
    fold_enrichment: np.ndarray,
    log2_fold_enrichment: np.ndarray,
    empirical_p_values: np.ndarray,
    fdr_matrix: np.ndarray,
    significant_mask: np.ndarray,
    type_counts: dict[str, int],
    number_of_permutations: int,
) -> pd.DataFrame:
    """Build ranked pair-level statistics for upper-triangle cell-type pairs."""

    records: list[dict[str, object]] = []
    n_types = len(tested_types)
    for i in range(n_types):
        for j in range(i + 1, n_types):
            type_i = tested_types[i]
            type_j = tested_types[j]
            records.append(
                {
                    "cell_type_i": type_i,
                    "cell_type_j": type_j,
                    "observed_contacts": int(observed_counts[i, j]),
                    "expected_contacts": float(expected_counts[i, j]),
                    "fold_enrichment": float(fold_enrichment[i, j]),
                    "log2_fold_enrichment": float(log2_fold_enrichment[i, j]),
                    "empirical_p_value": float(empirical_p_values[i, j]),
                    "fdr": float(fdr_matrix[i, j]),
                    "is_significant": bool(significant_mask[i, j]),
                    "cell_count_i": int(type_counts[type_i]),
                    "cell_count_j": int(type_counts[type_j]),
                    "number_of_permutations": int(number_of_permutations),
                }
            )

    if not records:
        return _empty_pair_statistics_dataframe()

    dataframe = pd.DataFrame.from_records(records)
    return dataframe.sort_values(
        by=["fdr", "fold_enrichment", "observed_contacts"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def _empty_pair_statistics_dataframe() -> pd.DataFrame:
    """Return an empty pair-statistics table with a fixed schema."""

    return pd.DataFrame(
        columns=[
            "cell_type_i",
            "cell_type_j",
            "observed_contacts",
            "expected_contacts",
            "fold_enrichment",
            "log2_fold_enrichment",
            "empirical_p_value",
            "fdr",
            "is_significant",
            "cell_count_i",
            "cell_count_j",
            "number_of_permutations",
        ]
    )
