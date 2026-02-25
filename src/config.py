from __future__ import annotations

from dataclasses import dataclass
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
    - the neighborhood radius for spatial domains
    - the number of spatial domains to infer with k-means
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
    neighborhood_radius: float = 50.0
    domain_n_clusters: int = 8

    @classmethod
    def from_dictionary(cls, data: dict[str, Any]) -> PipelineConfig:
        """Create from a raw dictionary (typically loaded from YAML)."""

        defaults = cls()
        return cls(
            minimum_counts=int(data.get("minimum_counts", defaults.minimum_counts)),
            maximum_counts_quantile=float(
                data.get("maximum_counts_quantile", defaults.maximum_counts_quantile)
            ),
            minimum_cells=int(data.get("minimum_cells", defaults.minimum_cells)),
            n_top_genes=int(data.get("n_top_genes", defaults.n_top_genes)),
            n_components=int(data.get("n_components", defaults.n_components)),
            leiden_resolution=float(
                data.get("leiden_resolution", defaults.leiden_resolution)
            ),
            rank_top_n=int(data.get("rank_top_n", defaults.rank_top_n)),
            minimum_logarithm_fold_change=float(
                data.get(
                    "minimum_logarithm_fold_change",
                    defaults.minimum_logarithm_fold_change,
                )
            ),
            maximum_adjusted_p_value=float(
                data.get("maximum_adjusted_p_value", defaults.maximum_adjusted_p_value)
            ),
            neighborhood_radius=float(
                data.get("neighborhood_radius", defaults.neighborhood_radius)
            ),
            domain_n_clusters=int(
                data.get("domain_n_clusters", defaults.domain_n_clusters)
            ),
        )


@dataclass(frozen=True)
class PlotsConfig:
    """Configuration for plotting behaviour.

    This bundles together:
    - the two genes to plot in transcript overlays
    """

    genes_to_plot: tuple[str, str] = ("TUBB", "CDH1")

    @classmethod
    def from_dictionary(cls, data: dict[str, Any]) -> PlotsConfig:
        """Create from a raw dictionary (typically loaded from YAML)."""

        gene_1, gene_2 = data["genes_to_plot"]
        return cls(genes_to_plot=(str(gene_1), str(gene_2)))


@dataclass
class Config:
    """Top-level configuration object used across the project.

    This bundles together:
    - all input paths (e.g. raw data directory)
    - all output paths (processed data, analysis, figures)
    - all tunable parameters loaded from the YAML file
    """

    raw_data_directories: tuple[Path, ...] = ()
    output_directory: Path | None = None
    processed_data_directory: Path | None = None
    results_directory: Path | None = None
    figures_directory: Path | None = None
    annotation_model: str = "llama3.1:8b"
    pipeline: PipelineConfig | None = None
    plots: PlotsConfig | None = None

    def load_from_yaml(self, configuration_path: Path) -> None:
        """Load configuration from a YAML file and populate this instance."""

        with configuration_path.open("r") as f:
            configuration = yaml.safe_load(f)

        raw_data_directories = tuple(
            Path(directory).resolve() for directory in configuration["data_directories"]
        )

        output_directory = Path(configuration["output_directory"]).resolve()

        self.raw_data_directories = raw_data_directories
        self.output_directory = output_directory
        self.processed_data_directory = output_directory / "processed"
        self.results_directory = output_directory / "analysis"
        self.figures_directory = output_directory / "figures"
        self.annotation_model = str(
            configuration.get("annotation_model", self.annotation_model)
        )
        self.pipeline = PipelineConfig.from_dictionary(configuration["pipeline"])
        self.plots = PlotsConfig.from_dictionary(configuration["plots"])

    def iterate_samples(self) -> tuple[tuple[str, Path], ...]:
        """Return (sample_id, directory_path) tuples in configured order."""

        return tuple(
            (raw_data_directory.name, raw_data_directory)
            for raw_data_directory in self.raw_data_directories
        )

    def create_directories(self) -> None:
        """Ensure all output directories exist."""

        for path in (
            self.processed_data_directory,
            self.results_directory,
            self.figures_directory,
        ):
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)
