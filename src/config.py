from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for analysis pipeline parameters.
    
    This bundles together:
    - the minimum number of counts per cell
    - the maximum counts quantile
    - the minimum number of cells per gene
    - the number of top genes to use for PCA
    - the number of components to use for PCA
    - the Leiden resolution parameter
    - the number of top genes to use for enrichment analysis
    - the minimum log fold change for enrichment analysis
    - the maximum adjusted p-value for enrichment analysis
    """

    minimum_counts: int = 200
    maximum_counts_quantile: float = 0.99
    minimum_cells: int = 100
    n_top_genes: int = 2000
    n_components: int = 30
    leiden_resolution: float = 0.5
    rank_top_n: int = 30
    minimum_logarithm_fold_change: float = 0.5
    maximum_adjusted_p_value: float = 0.05

    @classmethod
    def from_dictionary(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Create from a raw dictionary (typically loaded from YAML)."""

        return cls(
            minimum_counts=int(data["minimum_counts"]),
            maximum_counts_quantile=float(data["maximum_counts_quantile"]),
            minimum_cells=int(data["minimum_cells"]),
            n_top_genes=int(data["n_top_genes"]),
            n_components=int(data["n_components"]),
            leiden_resolution=float(data["leiden_resolution"]),
            rank_top_n=int(data["rank_top_n"]),
            minimum_logarithm_fold_change=float(data["minimum_logarithm_fold_change"]),
            maximum_adjusted_p_value=float(data["maximum_adjusted_p_value"]),
        )


@dataclass(frozen=True)
class PlotsConfig:
    """Configuration for plotting behaviour.

    This bundles together:
    - whether to plot cell boundaries
    - whether to plot transcripts
    - the genes to plot
    """

    plot_boundaries: bool = False
    plot_transcripts: bool = False
    genes_to_plot: tuple[str, ...] = ()

    @classmethod
    def from_dictionary(cls, data: dict[str, Any]) -> "PlotsConfig":
        """Create from a raw dictionary (typically loaded from YAML)."""
        
        return cls(
            plot_boundaries=data["plot_boundaries"],
            plot_transcripts=data["plot_transcripts"],
            genes_to_plot=tuple[str, ...](data["genes_to_plot"]),
        )


@dataclass
class Config:
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
    pipeline: PipelineConfig | None = None
    plots: PlotsConfig | None = None


    def load_from_yaml(self, configuration_path: Path) -> None:
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
        self.pipeline = PipelineConfig.from_dictionary(configuration["pipeline"])
        self.plots = PlotsConfig.from_dictionary(configuration["plots"])


    def create_directories(self) -> None:
        """Ensure all output directories exist. Raises ValueError if configuration is not loaded."""

        for path in (
            self.processed_data_directory,
            self.results_directory,
            self.figures_directory,
        ):
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)