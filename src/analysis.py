from __future__ import annotations

from anndata import AnnData
import pandas as pd
import scanpy as sc


def run_clustering(
    annotated_data: AnnData,
    n_components: int,
    leiden_resolution: float,
    metric: str = "cosine",
    leiden_flavor: str = "igraph",
) -> None:
    """Run clustering on the annotated data."""

    sc.pp.pca(annotated_data, n_components)
    sc.pp.neighbors(annotated_data, metric=metric)
    sc.tl.leiden(annotated_data, flavor=leiden_flavor, resolution=leiden_resolution)
    sc.tl.dendrogram(annotated_data, groupby="leiden")


def run_umap(annotated_data: AnnData) -> None:
    """Run UMAP on the annotated data."""

    sc.tl.umap(annotated_data)


def rank_genes(
    annotated_data: AnnData,
    groupby: str = "leiden",
    layer: str = "log_normalized",
) -> None:
    """Rank genes on the annotated data."""

    sc.tl.rank_genes_groups(annotated_data, groupby=groupby, layer=layer)


def compute_enriched_genes(
    annotated_data: AnnData,
    clusters: list[str],
    top_n: int,
    minimum_logarithm_fold_change: float,
    maximum_adjusted_p_value: float,
) -> dict[str, list[str]]:
    """Compute enriched genes on the annotated data."""

    gene_lists_by_cluster: dict[str, list[str]] = {}

    for cluster in clusters:
        cluster_dataframe = sc.get.rank_genes_groups_df(
            annotated_data, group=cluster
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


def build_domain_signatures(
    annotated_data: AnnData,
    *,
    domain_key: str = "spatial_domain",
    composition_key: str = "neighborhood_composition",
    composition_labels_key: str = "neighborhood_composition_labels",
    top_n: int = 6,
) -> dict[str, list[str]]:
    """Summarize each spatial domain by dominant neighborhood components."""

    if composition_key not in annotated_data.obsm:
        raise KeyError(
            f"Missing composition key `{composition_key}` in `annotated_data.obsm`."
        )
    if composition_labels_key not in annotated_data.uns:
        raise KeyError(
            f"Missing composition label key `{composition_labels_key}` in `annotated_data.uns`."
        )
    if domain_key not in annotated_data.obs:
        raise KeyError(f"Missing domain key `{domain_key}` in `annotated_data.obs`.")

    component_labels = [
        str(label) for label in annotated_data.uns[composition_labels_key]
    ]
    composition_dataframe = pd.DataFrame(
        annotated_data.obsm[composition_key],
        index=annotated_data.obs_names,
        columns=component_labels,
    )
    domain_values = annotated_data.obs[domain_key].astype(str)
    domain_means = composition_dataframe.groupby(domain_values, sort=True).mean()

    signatures: dict[str, list[str]] = {}
    for domain_id, row in domain_means.iterrows():
        top_components = row.sort_values(ascending=False).head(top_n).index.tolist()
        signatures[str(domain_id)] = [str(component) for component in top_components]
    return signatures
