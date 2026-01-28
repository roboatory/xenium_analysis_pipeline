from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from . import analysis
from .config import Config


def detect_spatial_keys(spatial_data):
    cell_boundary_key = (
        "cell_boundaries" if "cell_boundaries" in spatial_data.shapes else list(spatial_data.shapes.keys())[0]
    )
    nucleus_boundary_key = "nucleus_boundaries" if "nucleus_boundaries" in spatial_data.shapes else None
    transcripts_key = "transcripts" if "transcripts" in getattr(spatial_data, "points", {}) else None
    return cell_boundary_key, nucleus_boundary_key, transcripts_key


def _resolve_out_path(
    out_path: str | Path | None,
    config: Optional[Config],
    default_name: str,
) -> Path:
    if out_path is None:
        if config is None:
            raise ValueError("out_path or config must be provided.")
        if config.figures_directory is None:
            raise ValueError("Configuration not loaded. Call load_from_yaml() first.")
        out_path = config.figures_directory / default_name
    return Path(os.path.expanduser(str(out_path)))


def plot_cell_nucleus_boundaries(
    spatial_data,
    cell_key: str,
    nucleus_key: str | None,
    out_path: str | Path | None = None,
    config: Optional[Config] = None,
    figsize=(30, 30),
    dpi: int = 300,
) -> Path:
    figure, axes = plt.subplots(figsize=figsize, dpi=dpi)
    spatial_data.shapes[cell_key].plot(ax=axes, facecolor="none", edgecolor="black", linewidth=0.2)
    if nucleus_key is not None:
        spatial_data.shapes[nucleus_key].plot(ax=axes, facecolor="none", edgecolor="#1f77b4", linewidth=0.2)

    axes.set_title("Cell and nucleus boundaries")
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    axes.invert_yaxis()

    out_path = _resolve_out_path(
        out_path,
        config,
        default_name="xenium_cell_nucleus_boundaries.png",
    )
    figure.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(figure)
    return out_path


def plot_transcripts_with_boundaries(
    spatial_data,
    cell_key: str,
    nucleus_key: str | None,
    transcripts_key: str,
    genes_to_plot: list[str],
    out_path: str | Path | None = None,
    config: Optional[Config] = None,
    max_points: int = 50_000,
    figsize=(30, 30),
    dpi: int = 300,
) -> Path:
    transcripts_dataframe = spatial_data.points[transcripts_key]
    if hasattr(transcripts_dataframe, "compute"):
        transcripts_dataframe = transcripts_dataframe.compute()

    if "feature_name" not in transcripts_dataframe.columns:
        raise ValueError("Expected 'feature_name' column in transcripts")

    if len(genes_to_plot) < 2:
        raise ValueError("Please provide at least two genes in genes_to_plot")

    transcripts_plot_dataframe = transcripts_dataframe.loc[
        transcripts_dataframe["feature_name"].isin(genes_to_plot), ["x", "y", "feature_name"]
    ]
    if len(transcripts_plot_dataframe) > max_points:
        transcripts_plot_dataframe = transcripts_plot_dataframe.sample(n=max_points, random_state=0)

    figure, axes = plt.subplots(figsize=figsize, dpi=dpi)
    spatial_data.shapes[cell_key].plot(ax=axes, facecolor="none", edgecolor="black", linewidth=0.2)
    if nucleus_key is not None:
        spatial_data.shapes[nucleus_key].plot(ax=axes, facecolor="none", edgecolor="#1f77b4", linewidth=0.2)

    color_map = dict(zip(genes_to_plot, plt.cm.tab10.colors[: len(genes_to_plot)]))
    colors = transcripts_plot_dataframe["feature_name"].map(color_map)
    axes.scatter(
        transcripts_plot_dataframe["x"],
        transcripts_plot_dataframe["y"],
        s=2,
        alpha=1.0,
        linewidths=0.3,
        edgecolor="black",
        c=colors,
        rasterized=True,
    )

    for gene in genes_to_plot:
        gene_transcripts = transcripts_plot_dataframe[transcripts_plot_dataframe["feature_name"] == gene]
        if not gene_transcripts.empty:
            median_x = gene_transcripts["x"].median()
            median_y = gene_transcripts["y"].median()
            axes.text(
                median_x,
                median_y,
                gene,
                color=color_map[gene],
                fontsize=28,
                weight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.3"),
                ha="center",
                va="center",
            )

    axes.set_title("Transcripts (2+ genes) with Cell/Nucleus Boundaries", fontsize=34)
    axes.set_xlabel("X position", fontsize=24)
    axes.set_ylabel("Y position", fontsize=24)
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    axes.invert_yaxis()

    out_path = _resolve_out_path(
        out_path,
        config,
        default_name=f"xenium_cell_transcripts_{genes_to_plot}.png",
    )
    figure.savefig(out_path, bbox_inches="tight", dpi=600)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color_map[gene],
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=8,
        )
        for gene in genes_to_plot
    ]
    axes.legend(
        legend_handles,
        genes_to_plot,
        markerscale=1,
        frameon=False,
        fontsize=20,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Genes",
        title_fontsize=24,
        borderaxespad=0.0,
    )
    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.close(figure)

    return out_path


def plot_qc_hist(
    annotated_data,
    cutoffs: list[int],
    cutoff_colors: list[str],
    out_path: str | Path | None = None,
    config: Optional[Config] = None,
    zoom_max: int = 2500,
) -> Path:
    figure, axes_array = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [2, 1]})

    axes_array[0].hist(annotated_data.obs["total_counts"], bins=60, color="black", alpha=0.8)
    axes_array[0].set_xlabel("Transcripts per cell")
    axes_array[0].set_ylabel("Number of cells")
    axes_array[0].set_title("QC: transcripts per cell (full range)")

    for index, (cutoff, color) in enumerate(zip(cutoffs, cutoff_colors)):
        axes_array[0].axvline(cutoff, color=color, linestyle="--", linewidth=2, alpha=0.8)
        axes_array[0].text(
            cutoff,
            axes_array[0].get_ylim()[1] * (0.9 - 0.05 * index),
            f"cutoff={cutoff}",
            color=color,
            fontsize=14,
            ha="left",
            va="top",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=0.2),
        )

    counts_zoom = annotated_data.obs["total_counts"][annotated_data.obs["total_counts"] <= zoom_max]
    axes_array[1].hist(counts_zoom, bins=50, color="dodgerblue", alpha=0.9)
    axes_array[1].set_xlabel("Transcripts per cell\n(zoom 0-2500)")
    axes_array[1].set_ylabel("Number of cells")
    axes_array[1].set_title("Zoom: 0-2500 transcripts")
    axes_array[1].set_xlim(0, zoom_max)
    axes_array[1].set_yticks([])

    for index, (cutoff, color) in enumerate(zip(cutoffs, cutoff_colors)):
        if cutoff <= zoom_max:
            axes_array[1].axvline(cutoff, color=color, linestyle="--", linewidth=2, alpha=0.8)
            axes_array[1].text(
                cutoff,
                axes_array[1].get_ylim()[1] * (0.92 - 0.07 * index),
                f"cutoff={cutoff}",
                color=color,
                fontsize=13,
                ha="left",
                va="top",
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=0.2),
            )

    plt.tight_layout()

    out_path = _resolve_out_path(
        out_path,
        config,
        default_name="xenium_transcripts_per_cell.png",
    )
    figure.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(figure)
    return out_path


def plot_umap_leiden(annotated_data, out_path: str | Path | None = None, config: Optional[Config] = None) -> Path:
    umap_figure = sc.pl.umap(annotated_data, color="leiden", show=False, return_fig=True)
    out_path = _resolve_out_path(out_path, config, default_name="umap_leiden.png")
    umap_figure.savefig(out_path, bbox_inches="tight", dpi=600)
    return out_path


def _get_spatial_coords(annotated_data):
    if "spatial" in annotated_data.obsm:
        coordinates = annotated_data.obsm["spatial"]
        x_coordinates = coordinates[:, 0]
        y_coordinates = coordinates[:, 1]
    else:
        possible_coordinate_columns = [
            ("x_centroid", "y_centroid"),
            ("center_x", "center_y"),
            ("x", "y"),
        ]
        for x_column_name, y_column_name in possible_coordinate_columns:
            if x_column_name in annotated_data.obs.columns and y_column_name in annotated_data.obs.columns:
                x_coordinates = annotated_data.obs[x_column_name].to_numpy()
                y_coordinates = annotated_data.obs[y_column_name].to_numpy()
                break
        else:
            raise ValueError("No spatial coordinates found in annotated_data.obsm['spatial'] or known columns.")
    return x_coordinates, y_coordinates


def plot_cluster_overlay(
    annotated_data,
    annotations: dict,
    out_path: str | Path | None = None,
    config: Optional[Config] = None,
) -> Path:
    x_coordinates, y_coordinates = _get_spatial_coords(annotated_data)

    leiden = annotated_data.obs["leiden"].astype(str)
    ordered_categories, cluster_codes = analysis.ordered_clusters(leiden)

    color_map = mpl.colormaps.get_cmap("tab20")
    colors = np.array([color_map(code % 20) for code in cluster_codes.codes])

    figure, axes = plt.subplots(figsize=(20, 15), dpi=300)
    axes.scatter(
        x_coordinates,
        y_coordinates,
        s=2,
        c=colors,
        alpha=0.9,
        linewidths=0,
        rasterized=True,
    )

    legend_handles = []
    for index, category in enumerate(cluster_codes.categories):
        cell_type = "unknown"
        if str(category) in annotations and isinstance(annotations[str(category)], dict):
            cell_type = annotations[str(category)].get("annotation", "unknown") or "unknown"
        label = f"{category}: {cell_type}"
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=color_map(index % 20),
                markeredgecolor="none",
                label=label,
            )
        )

    axes.legend(
        handles=legend_handles,
        title="Cell type",
        frameon=False,
        fontsize=10,
        title_fontsize=12,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    axes.set_title("Cell type annotations (dot plot)", fontsize=24)
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    axes.invert_yaxis()
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    out_path = _resolve_out_path(out_path, config, default_name="xenium_celltype_overlay.png")
    figure.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(figure)
    return out_path


def plot_rank_genes_dotplot(
    annotated_data,
    out_path: str | Path | None = None,
    config: Optional[Config] = None,
    n_genes: int = 5,
) -> Path:
    dotplot_figure = sc.pl.rank_genes_groups_dotplot(annotated_data, n_genes=n_genes, show=False, return_fig=True)
    out_path = _resolve_out_path(out_path, config, default_name="xenium_rank_genes_dotplot.png")
    dotplot_figure.savefig(out_path, bbox_inches="tight", dpi=600)
    return out_path


def plot_celltype_pair(
    annotated_data,
    celltype_a: str,
    celltype_b: str,
    label_col: str = "cell_type",
    match_mode: str = "substring",
    colors=("crimson", "steelblue"),
    point_size: int = 2,
    alpha: float = 0.9,
    out_path: str | Path | None = None,
    config: Optional[Config] = None,
) -> Path:
    x_coordinates, y_coordinates = _get_spatial_coords(annotated_data)

    labels = annotated_data.obs[label_col].astype(str)
    if match_mode == "exact":
        mask_a = labels == celltype_a
        mask_b = labels == celltype_b
    else:
        mask_a = labels.str.contains(celltype_a, case=False, regex=False)
        mask_b = labels.str.contains(celltype_b, case=False, regex=False)

    figure, axes = plt.subplots(figsize=(20, 15), dpi=300)
    axes.scatter(
        x_coordinates[mask_a],
        y_coordinates[mask_a],
        s=point_size,
        c=colors[0],
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )
    axes.scatter(
        x_coordinates[mask_b],
        y_coordinates[mask_b],
        s=point_size,
        c=colors[1],
        alpha=alpha,
        linewidths=0,
        rasterized=True,
    )

    axes.set_title(f"{celltype_a} vs {celltype_b}", fontsize=24)
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    axes.invert_yaxis()

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=colors[0],
            label=celltype_a,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=colors[1],
            label=celltype_b,
        ),
    ]
    axes.legend(legend_handles, frameon=False, fontsize=12, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if out_path is None:
        safe_celltype_a = celltype_a.replace(" ", "_").lower()
        safe_celltype_b = celltype_b.replace(" ", "_").lower()
        out_path = _resolve_out_path(
            None,
            config,
            default_name=f"xenium_{safe_celltype_a}_vs_{safe_celltype_b}.png",
        )
    else:
        out_path = _resolve_out_path(out_path, config, default_name="")
    figure.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(figure)
    return out_path
