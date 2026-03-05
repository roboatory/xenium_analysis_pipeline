from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfiguration:
    """Configuration for analysis pipeline parameters.

    This bundles together:
    - the minimum number of counts per cell
    - the maximum counts quantile
    - the minimum number of cells per gene
    - the number of principal components for PCA
    - the neighborhood radius for spatial domains
    - the radius for cell-type contact colocalization
    - the number of spatial domains to infer with k-means
    - the number of top genes to use for enrichment analysis
    - the minimum log fold change for enrichment analysis
    - the maximum adjusted p-value for enrichment analysis
    """

    minimum_counts: int
    maximum_counts_quantile: float
    minimum_cells: int
    pca_n_components: int
    neighborhood_radius: float
    colocalization_radius: float
    domain_n_clusters: int
    rank_top_n: int
    minimum_logarithm_fold_change: float
    maximum_adjusted_p_value: float

    @classmethod
    def from_dictionary(
        cls: type[PipelineConfiguration],
        data: dict[str, Any],
    ) -> PipelineConfiguration:
        """Create from a raw dictionary (typically loaded from YAML)."""

        return cls(
            minimum_counts=int(data["minimum_counts"]),
            maximum_counts_quantile=float(data["maximum_counts_quantile"]),
            minimum_cells=int(data["minimum_cells"]),
            pca_n_components=int(data["pca_n_components"]),
            neighborhood_radius=float(data["neighborhood_radius"]),
            colocalization_radius=float(data.get("colocalization_radius", 20.0)),
            domain_n_clusters=int(data["domain_n_clusters"]),
            rank_top_n=int(data["rank_top_n"]),
            minimum_logarithm_fold_change=float(data["minimum_logarithm_fold_change"]),
            maximum_adjusted_p_value=float(data["maximum_adjusted_p_value"]),
        )


@dataclass(frozen=True)
class PlotsConfiguration:
    """Configuration for plotting behaviour.

    This bundles together the genes to plot for transcript figures.
    """

    genes_to_plot: tuple[str, ...] = ()

    @classmethod
    def from_dictionary(
        cls: type[PlotsConfiguration],
        data: dict[str, Any],
    ) -> PlotsConfiguration:
        """Create from a raw dictionary (typically loaded from YAML)."""

        return cls(
            genes_to_plot=tuple(data["genes_to_plot"]),
        )


@dataclass
class Configuration:
    """Top-level configuration object used across the project.

    This bundles together:
    - all input paths (e.g. raw data directory)
    - all output paths (processed data, analysis, figures)
    - all tunable parameters loaded from the YAML file
    """

    raw_data_directory: Path | None = None
    output_directory: Path | None = None
    processed_data_directory: Path | None = None
    results_directory: Path | None = None
    figures_directory: Path | None = None
    annotation_model: str = "llama3.1:8b"
    pipeline: PipelineConfiguration | None = None
    plots: PlotsConfiguration | None = None

    def load_from_yaml(
        self: type[Configuration],
        configuration_path: Path,
    ) -> None:
        """Load configuration from a YAML file and populate this instance."""

        with configuration_path.open("r") as f:
            configuration = yaml.safe_load(f)

        raw_data_directory = Path(configuration["data_directory"]).resolve()
        output_directory = Path(configuration["output_directory"]).resolve()

        self.raw_data_directory = raw_data_directory
        self.output_directory = output_directory
        self.processed_data_directory = output_directory / "processed"
        self.results_directory = output_directory / "analysis"
        self.figures_directory = output_directory / "figures"
        self.annotation_model = str(
            configuration.get("annotation_model", self.annotation_model)
        )
        self.pipeline = PipelineConfiguration.from_dictionary(configuration["pipeline"])
        self.plots = PlotsConfiguration.from_dictionary(configuration["plots"])

    def create_directories(
        self: type[Configuration],
    ) -> None:
        """Ensure all output directories exist."""

        for path in (
            self.processed_data_directory,
            self.results_directory,
            self.figures_directory,
        ):
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)
