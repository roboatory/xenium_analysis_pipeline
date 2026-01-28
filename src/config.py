from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for analysis pipeline parameters."""

    min_counts: int = 200
    max_counts_quantile: float = 0.99
    min_cells: int = 100
    n_top_genes: int = 2000
    n_comps: int = 30
    leiden_resolution: float = 0.5
    rank_top_n: int = 30
    min_logfc: float = 0.5
    max_adj_pval: float = 0.05

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Create from a raw dictionary (typically loaded from YAML)."""

        return cls(
            min_counts=int(data.get("min_counts", 200)),
            max_counts_quantile=float(data.get("max_counts_quantile", 0.99)),
            min_cells=int(data.get("min_cells", 100)),
            n_top_genes=int(data.get("n_top_genes", 2000)),
            n_comps=int(data.get("n_comps", 30)),
            leiden_resolution=float(data.get("leiden_resolution", 0.5)),
            rank_top_n=int(data.get("rank_top_n", 30)),
            min_logfc=float(data.get("min_logfc", 0.5)),
            max_adj_pval=float(data.get("max_adj_pval", 0.05)),
        )


@dataclass(frozen=True)
class PlotsConfig:
    """Configuration for plotting behaviour."""

    plot_boundaries: bool = False
    plot_transcripts: bool = False
    # Use an immutable collection to keep Config hashable/frozen-friendly
    genes_to_plot: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlotsConfig":
        """Create from a raw dictionary (typically loaded from YAML)."""

        plot_boundaries = bool(data.get("plot_boundaries", False))
        plot_transcripts = bool(data.get("plot_transcripts", False))

        raw_genes = data.get("genes_to_plot", [])
        genes: list[str]
        if isinstance(raw_genes, str):
            genes = [gene.strip() for gene in raw_genes.split(",") if gene.strip()]
        elif isinstance(raw_genes, (list, tuple)):
            genes = [str(gene).strip() for gene in raw_genes if str(gene).strip()]
        else:
            genes = []

        return cls(
            plot_boundaries=plot_boundaries,
            plot_transcripts=plot_transcripts,
            genes_to_plot=tuple(genes),
        )


@dataclass(frozen=True)
class Config:
    """Top‑level configuration object used across the project.

    This bundles together:
    - all input paths (e.g. raw data directory)
    - all output paths (processed data, analysis, figures)
    - all tunable parameters loaded from the YAML file
    """

    project_root: Path
    raw_data_dir: Path
    outputs_root: Path
    processed_data_dir: Path
    results_dir: Path
    figures_dir: Path
    pipeline: PipelineConfig
    plots: PlotsConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any], config_dir: Path) -> "Config":
        """Create a fully-populated Config from raw YAML data."""

        project_root = Path(data.get("project_root", ".")).expanduser().resolve()

        # Input data directory (absolute or relative)
        raw_data_dir = Path(data["data_dir"]).expanduser().resolve()

        # Root directory for all outputs (absolute or relative to project root)
        outputs_dir_value = data.get("outputs_dir", "../outputs")
        outputs_root = cls._resolve_directory(project_root, outputs_dir_value)

        processed_data_dir = outputs_root / "processed"
        results_dir = outputs_root / "analysis"
        figures_dir = outputs_root / "figures"

        pipeline = PipelineConfig.from_dict(data.get("pipeline", {}))
        plots = PlotsConfig.from_dict(data.get("plots", {}))

        return cls(
            project_root=project_root,
            raw_data_dir=raw_data_dir,
            outputs_root=outputs_root,
            processed_data_dir=processed_data_dir,
            results_dir=results_dir,
            figures_dir=figures_dir,
            pipeline=pipeline,
            plots=plots,
        )

    @staticmethod
    def _resolve_directory(project_root: Path, relative_or_absolute_path: str) -> Path:
        """Resolve a directory, allowing either absolute or project‑relative paths."""

        candidate_path = Path(relative_or_absolute_path).expanduser()
        return candidate_path if candidate_path.is_absolute() else project_root / candidate_path

    def ensure_dirs(self) -> None:
        for path in self._dir_paths():
            path.mkdir(parents=True, exist_ok=True)

    def _dir_paths(self) -> Iterable[Path]:
        return (
            self.raw_data_dir,
            self.processed_data_dir,
            self.results_dir,
            self.figures_dir,
        )
