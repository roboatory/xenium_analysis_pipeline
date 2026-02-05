from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

from .config import Config

from spatialdata import SpatialData
import spatialdata_plot  # noqa: F401

from anndata import AnnData


def plot_cell_and_nucleus_boundaries(
    spatial_data: SpatialData,
    configuration: Config,
    cell_key: str = "cell_boundaries",
    nucleus_key: str = "nucleus_boundaries",
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
                element=cell_key,
                fill_alpha=0.0,
                outline_color="#2d2d2d",
                outline_width=0.2,
                outline_alpha=1.0,
            )
            .pl.render_shapes(
                element=nucleus_key,
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
    spatial_data: SpatialData,
    configuration: Config,
    genes: list[str],
    palette: list[str],
    points_key: str = "transcripts",
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
                element=points_key,
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
    annotated_data,
    cutoffs: list[float],
    cutoff_colors: list[str],
    configuration: Config,
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
    spatial_data,
    configuration: Config,
    out_path: str | Path | None = None,
    table_key: str = "table",
):
    """Plot UMAP colored by Leiden clusters."""

    out_path = configuration.figures_directory / "umap_leiden.png"
    annotated_data = spatial_data[table_key]

    _show = plt.show
    plt.show = lambda: None
    try:
        _dpi = 600
        umap_figure = sc.pl.umap(
            annotated_data, color="leiden", show=False, return_fig=True
        )
        umap_figure.savefig(out_path, bbox_inches="tight", dpi=_dpi)
        plt.close(umap_figure)
    finally:
        plt.show = _show


def plot_cluster_overlay(
    spatial_data: SpatialData,
    configuration: Config,
    cell_key: str = "cell_boundaries",
    table_key: str = "table",
    cluster_key: str = "leiden",
) -> None:
    """Plot cell shapes colored by cluster labels (e.g. Leiden)."""

    out_path = configuration.figures_directory / f"xenium_{cluster_key}_overlay.png"
    table = spatial_data.tables[table_key]
    gdf = spatial_data.shapes[cell_key].copy()

    if "cell_id" in table.obs.columns:
        cell_to_cluster = table.obs.set_index("cell_id")[cluster_key]
    else:
        cell_to_cluster = table.obs[cluster_key]
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
        fig.savefig(out_path, bbox_inches="tight", dpi=_dpi)
        plt.close(fig)
    finally:
        plt.show = _show


def plot_rank_genes_dotplot(
    annotated_data: AnnData,
    configuration: Config,
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
