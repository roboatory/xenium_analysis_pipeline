from __future__ import annotations


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

from .config import Configuration

from spatialdata import SpatialData
import spatialdata_plot  # noqa: F401

from anndata import AnnData
import pandas as pd


def plot_cell_and_nucleus_boundaries(
    configuration: Configuration,
    spatial_data: SpatialData,
) -> None:
    """Plot cell and nucleus boundaries."""

    out_path = configuration.figures_directory / "xenium_cell_nucleus_boundaries.png"

    _show = plt.show
    plt.show = lambda: None
    try:
        _dpi = 600
        fig, ax = plt.subplots(figsize=(14, 14), dpi=_dpi)

        (
            spatial_data.pl.render_shapes(
                element="cell_boundaries",
                fill_alpha=0.0,
                outline_color="#2d2d2d",
                outline_width=0.2,
                outline_alpha=1.0,
            )
            .pl.render_shapes(
                element="nucleus_boundaries",
                fill_alpha=0.0,
                outline_color="#e64b35",
                outline_width=0.35,
                outline_alpha=1.0,
            )
            .pl.show(ax=ax, title="", dpi=_dpi)
        )

        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight", dpi=_dpi)
        plt.close(fig)
    finally:
        plt.show = _show


def plot_transcripts(
    configuration: Configuration,
    spatial_data: SpatialData,
    genes: list[str],
    palette: list[str],
    max_points: int = 50_000,
) -> None:
    """Plot transcripts."""

    out_path = (
        configuration.figures_directory / f"xenium_transcripts_{'_'.join(genes)}.png"
    )

    _show = plt.show
    plt.show = lambda: None
    try:
        _dpi = 600
        fig, ax = plt.subplots(figsize=(14, 14), dpi=_dpi)

        (
            spatial_data.pl.render_points(
                element="transcripts",
                color="feature_name",
                groups=genes,
                size=2,
                max_points=max_points,
                palette=palette,
            ).pl.show(ax=ax, title="", dpi=_dpi, colorbar=False)
        )

        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight", dpi=_dpi)
        plt.close(fig)
    finally:
        plt.show = _show


def plot_qc_histogram(
    configuration: Configuration,
    annotated_data: AnnData,
    cutoffs: list[float],
    cutoff_colors: list[str],
) -> None:
    """Plot QC histogram."""

    out_path = configuration.figures_directory / "xenium_transcripts_per_cell.png"

    cutoffs = [
        int(cutoffs[0]),
        int(np.quantile(annotated_data.obs["total_counts"], cutoffs[1])),
    ]

    _show = plt.show
    plt.show = lambda: None
    try:
        _dpi = 600
        figure, ax = plt.subplots(figsize=(10, 5), dpi=_dpi)
        ax.hist(annotated_data.obs["total_counts"], bins=100, color="black", alpha=0.8)
        ax.set_xlabel("transcripts per cell")
        ax.set_ylabel("number of cells")
        ax.set_title("QC: transcripts per cell (full range)")

        for index, (cutoff, color) in enumerate(zip(cutoffs, cutoff_colors)):
            ax.axvline(cutoff, color=color, linestyle="--", linewidth=2, alpha=0.8)
            ax.text(
                cutoff,
                ax.get_ylim()[1] * (0.9 - 0.05 * index),
                f"cutoff={cutoff}",
                color=color,
                fontsize=12,
                ha="left",
                va="top",
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=0.2),
            )

        figure.savefig(out_path, bbox_inches="tight", dpi=_dpi)
        plt.close(figure)
    finally:
        plt.show = _show


def plot_umap_leiden(
    configuration: Configuration,
    spatial_data: SpatialData,
) -> None:
    """Plot UMAP colored by Leiden clusters."""

    out_path = configuration.figures_directory / "umap_leiden.png"
    annotated_data = spatial_data["table"]

    _show = plt.show
    plt.show = lambda: None
    try:
        _dpi = 600
        umap_figure = sc.pl.umap(
            annotated_data,
            color="cell_type",
            show=False,
            return_fig=True,
        )
        umap_figure.savefig(out_path, bbox_inches="tight", dpi=_dpi)
        plt.close(umap_figure)
    finally:
        plt.show = _show


def plot_cluster_overlay(
    configuration: Configuration,
    spatial_data: SpatialData,
    cluster_key: str = "cell_type",
) -> None:
    """Plot cell shapes colored by cluster labels (e.g. Leiden)."""

    out_path = configuration.figures_directory / f"xenium_{cluster_key}_overlay.png"
    table = spatial_data.tables["table"]
    gdf = spatial_data.shapes["cell_boundaries"].copy()

    cell_to_cluster = table.obs.set_index("cell_id")[cluster_key]
    gdf[cluster_key] = gdf.index.map(cell_to_cluster)
    gdf = gdf.dropna(subset=[cluster_key])

    _show = plt.show
    plt.show = lambda: None
    try:
        _dpi = 600
        fig, ax = plt.subplots(figsize=(14, 14), dpi=_dpi)
        n_clusters = max(gdf[cluster_key].nunique(), 1)
        cmap = matplotlib.colormaps["Set3"].resampled(n_clusters)
        gdf.plot(
            column=cluster_key,
            ax=ax,
            categorical=True,
            legend=True,
            edgecolor="gray",
            linewidth=0.15,
            cmap=cmap,
            legend_kwds={
                "loc": "center left",
                "bbox_to_anchor": (1, 0.5),
                "fontsize": 9,
            },
        )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.invert_yaxis()
        fig.savefig(out_path, bbox_inches="tight", dpi=_dpi)
        plt.close(fig)
    finally:
        plt.show = _show


def plot_rank_genes_dotplot(
    configuration: Configuration,
    annotated_data: AnnData,
    n_genes: int = 5,
) -> None:
    """Plot rank genes groups dotplot."""

    out_path = configuration.figures_directory / f"rank_genes_dotplot_top_{n_genes}.png"

    _show = plt.show
    plt.show = lambda: None
    try:
        _dpi = 600
        dotplot = sc.pl.rank_genes_groups_dotplot(
            annotated_data,
            n_genes=n_genes,
            show=False,
            return_fig=True,
        )
        dotplot.make_figure()
        fig = dotplot.fig
        fig.savefig(out_path, bbox_inches="tight", dpi=_dpi)
        plt.close(fig)
    finally:
        plt.show = _show


def plot_colocalization_contact_counts(
    configuration: Configuration,
    counts: pd.DataFrame,
) -> None:
    """Plot heatmap of observed cell-type contact counts."""

    out_path = (
        configuration.figures_directory / "xenium_colocalization_contact_counts.png"
    )

    _show = plt.show
    plt.show = lambda: None
    try:
        _dpi = 600
        figure, axis = plt.subplots(figsize=(12, 10), dpi=_dpi)
        image = axis.imshow(np.log1p(counts.to_numpy()), cmap="magma")
        axis.set_xticks(np.arange(counts.shape[1]))
        axis.set_yticks(np.arange(counts.shape[0]))
        axis.set_xticklabels(counts.columns)
        axis.set_yticklabels(counts.index)
        axis.tick_params(axis="x", labelrotation=90)
        axis.set_xlabel("cell type")
        axis.set_ylabel("cell type")
        axis.set_title("Observed 1st-degree contacts (log1p counts)")
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label("log1p(contact count)")
        figure.tight_layout()
        figure.savefig(out_path, bbox_inches="tight", dpi=_dpi)
        plt.close(figure)
    finally:
        plt.show = _show


def plot_colocalization_contact_row_proportions(
    configuration: Configuration,
    proportions: pd.DataFrame,
) -> None:
    """Plot heatmap of row-normalized observed cell-type contact proportions."""

    out_path = (
        configuration.figures_directory
        / "xenium_colocalization_contact_row_proportions.png"
    )

    _show = plt.show
    plt.show = lambda: None
    try:
        _dpi = 600
        figure, axis = plt.subplots(figsize=(12, 10), dpi=_dpi)
        image = axis.imshow(proportions.to_numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_xticks(np.arange(proportions.shape[1]))
        axis.set_yticks(np.arange(proportions.shape[0]))
        axis.set_xticklabels(proportions.columns)
        axis.set_yticklabels(proportions.index)
        axis.tick_params(axis="x", labelrotation=90)
        axis.set_xlabel("cell type")
        axis.set_ylabel("cell type")
        axis.set_title("Observed 1st-degree contacts (row-normalized)")
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label("row-normalized contact proportion")
        figure.tight_layout()
        figure.savefig(out_path, bbox_inches="tight", dpi=_dpi)
        plt.close(figure)
    finally:
        plt.show = _show
