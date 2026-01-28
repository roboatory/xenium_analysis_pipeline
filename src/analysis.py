from __future__ import annotations

import pandas as pd
import scanpy as sc


def default_sort_key(cluster_value: str):
    return (0, int(cluster_value)) if cluster_value.isdigit() else (1, cluster_value)


def ordered_clusters(leiden_series, sort_key=default_sort_key):
    ordered_categories = sorted(pd.unique(leiden_series), key=sort_key)
    cluster_codes = pd.Categorical(leiden_series, categories=ordered_categories, ordered=True)
    return ordered_categories, cluster_codes


def run_clustering(
    annotated_data,
    n_comps: int = 30,
    metric: str = "cosine",
    leiden_flavor: str = "igraph",
    leiden_iterations: int = -1,
    leiden_resolution: float = 0.5,
) -> None:
    sc.pp.pca(annotated_data, n_comps=n_comps)
    sc.pp.neighbors(annotated_data, metric=metric)
    sc.tl.leiden(
        annotated_data,
        flavor=leiden_flavor,
        n_iterations=leiden_iterations,
        resolution=leiden_resolution,
    )


def run_umap(annotated_data) -> None:
    sc.tl.umap(annotated_data)


def rank_genes(
    annotated_data,
    groupby: str = "leiden",
    layer: str = "log_norm",
    show_points: bool = True,
) -> None:
    sc.tl.rank_genes_groups(annotated_data, groupby=groupby, layer=layer, pts=show_points)


def compute_enriched_genes(
    annotated_data,
    clusters: list[str],
    top_n: int = 30,
    min_logfc: float = 0.5,
    max_adj_pval: float = 0.05,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    gene_lists_by_cluster: dict[str, list[str]] = {}
    rows: list[dict[str, str]] = []

    for cluster in clusters:
        cluster_dataframe = sc.get.rank_genes_groups_df(annotated_data, group=cluster)
        cluster_dataframe = cluster_dataframe.dropna(subset=["names"])

        if "logfoldchanges" in cluster_dataframe.columns:
            cluster_dataframe = cluster_dataframe[cluster_dataframe["logfoldchanges"] >= min_logfc]
        if "pvals_adj" in cluster_dataframe.columns:
            cluster_dataframe = cluster_dataframe[cluster_dataframe["pvals_adj"] <= max_adj_pval]

        genes = cluster_dataframe["names"].astype(str).head(top_n).tolist()
        if len(genes) == 0:
            fallback_dataframe = sc.get.rank_genes_groups_df(annotated_data, group=cluster)
            genes = fallback_dataframe["names"].astype(str).head(top_n).tolist()

        gene_lists_by_cluster[str(cluster)] = genes
        rows.append({"cluster": str(cluster), "genes": ", ".join(genes)})

    return gene_lists_by_cluster, pd.DataFrame(rows)


def apply_celltype_annotations(annotated_data, annotations: dict) -> dict[str, str]:
    cluster_to_celltype = {
        str(cluster): (annotation.get("annotation") if isinstance(annotation, dict) else str(annotation))
        for cluster, annotation in annotations.items()
    }
    annotated_data.obs["cell_type"] = (
        annotated_data.obs["leiden"].astype(str).map(cluster_to_celltype).fillna("unknown")
    )
    return cluster_to_celltype
