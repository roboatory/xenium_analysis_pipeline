from __future__ import annotations

from anndata import AnnData
import pandas as pd
import scanpy as sc

from .logging_utils import get_logger

logger = get_logger(__name__)


def run_clustering(
    annotated_data: AnnData,
    pca_n_components: int,
) -> None:
    """Run clustering on the annotated data."""

    logger.debug("running PCA/neighbors/leiden with %s components", pca_n_components)
    sc.pp.pca(annotated_data, n_comps=pca_n_components)
    sc.pp.neighbors(annotated_data, metric="cosine")
    sc.tl.leiden(
        annotated_data,
        resolution=0.5,
        flavor="igraph",
    )
    logger.info(
        "found %s leiden clusters",
        annotated_data.obs["leiden"].nunique(),
    )


def run_umap(
    annotated_data: AnnData,
) -> None:
    """Run UMAP on the annotated data."""

    logger.debug("running UMAP embedding")
    sc.tl.umap(annotated_data)
    logger.debug("UMAP embedding complete")


def rank_genes(
    annotated_data: AnnData,
) -> None:
    """Rank genes on the annotated data."""

    logger.debug("ranking marker genes by leiden cluster")
    sc.tl.rank_genes_groups(
        annotated_data,
        groupby="leiden",
        layer="log_normalized",
        pts=True,
    )
    logger.debug("gene ranking complete")


def compute_enriched_genes(
    annotated_data: AnnData,
    top_n: int,
    minimum_logarithm_fold_change: float,
    maximum_adjusted_p_value: float,
) -> dict[str, list[str]]:
    """Compute enriched genes on the annotated data."""

    logger.debug(
        "collecting enriched genes with top_n=%s, minimum_logarithm_fold_change=%s, maximum_adjusted_p_value=%s",
        top_n,
        minimum_logarithm_fold_change,
        maximum_adjusted_p_value,
    )
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

    logger.info("computed marker lists for %s clusters", len(gene_lists_by_cluster))
    return gene_lists_by_cluster
