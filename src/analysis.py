from __future__ import annotations

from anndata import AnnData
import pandas as pd
import scanpy as sc


def run_clustering(
    annotated_data: AnnData,
    pca_n_components: int,
) -> None:
    """Run clustering on the annotated data."""

    sc.pp.pca(annotated_data, n_comps=pca_n_components)
    sc.pp.neighbors(annotated_data, metric="cosine")
    sc.tl.leiden(
        annotated_data,
        resolution=0.5,
        flavor="igraph",
    )


def run_umap(
    annotated_data: AnnData,
) -> None:
    """Run UMAP on the annotated data."""

    sc.tl.umap(annotated_data)


def rank_genes(
    annotated_data: AnnData,
) -> None:
    """Rank genes on the annotated data."""

    sc.tl.rank_genes_groups(
        annotated_data,
        groupby="leiden",
        layer="log_normalized",
        pts=True,
    )


def compute_enriched_genes(
    annotated_data: AnnData,
    top_n: int,
    minimum_logarithm_fold_change: float,
    maximum_adjusted_p_value: float,
) -> dict[str, list[str]]:
    """Compute enriched genes on the annotated data."""

    gene_lists_by_cluster: dict[str, list[str]] = {}

    for cluster in pd.unique(annotated_data.obs["leiden"]):
        cluster_dataframe = sc.get.rank_genes_groups_df(
            annotated_data,
            group=cluster,
        ).dropna(subset=["names"])

        cluster_dataframe = cluster_dataframe[
            cluster_dataframe["logfoldchanges"] >= minimum_logarithm_fold_change
        ]
        cluster_dataframe = cluster_dataframe[
            cluster_dataframe["pvals_adj"] <= maximum_adjusted_p_value
        ]

        genes = cluster_dataframe["names"].astype(str).head(top_n).tolist()

        gene_lists_by_cluster[str(cluster)] = genes

    return gene_lists_by_cluster
