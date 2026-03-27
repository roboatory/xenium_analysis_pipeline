# Prostate Cancer Spatial Transcriptomics Pipeline

This repository runs a Xenium-based spatial transcriptomics pipeline. The workflow has two required execution steps:

1. `uv run ingest.py`
2. `uv run main.py`

The first command ingests raw Xenium output into a processed `SpatialData` Zarr. The second command runs preprocessing, clustering, marker analysis, LLM-based annotation, spatial domain analysis, and colocalization.

## Prerequisites

Install `uv` by following Astral's official instructions: [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/). If you are new to `uv`, it is a fast Python package and environment manager; in this repository, `uv run ...` will create/use the project environment and run the scripts with the locked dependencies.

Install Ollama using the official docs: [Ollama quickstart](https://docs.ollama.com/quickstart) or [download page](https://ollama.com/download). This pipeline uses a local Ollama-compatible server for annotation, so Ollama must be installed locally and running before `main.py`.

## Required model and server

The annotation stage requires the `llama3.1:8b` model.

Pull the model once:

```bash
ollama pull llama3.1:8b
```

Start the local Ollama server in a separate terminal before running the pipeline:

```bash
ollama serve
```

The pipeline expects the Ollama API at `http://localhost:11434`, which is the default server address used by `src/annotation.py`.

## Configure paths

Edit [`config.yaml`](/Users/rohit/Desktop/prostate_cancer/code/config.yaml) so these paths match your machine:

- `data_directory`: path to the raw Xenium output directory
- `output_directory`: path where processed data, analysis artifacts, figures, and logs should be written

The default config also sets:

- `annotation_model: "llama3.1:8b"`

## Run the pipeline end-to-end

From the repository root, run the two required steps in order:

```bash
uv run ingest.py
uv run main.py
```

### Step 1: ingest raw data

```bash
uv run ingest.py
```

This reads the raw Xenium dataset from `data_directory` and writes:

- `processed/processed.zarr`

under the configured `output_directory`.

### Step 2: run the analysis pipeline

```bash
uv run main.py
```

This reads `processed/processed.zarr` and runs the downstream analysis stages:

- QC filtering and normalization
- PCA, neighbor graph construction, Leiden clustering, and UMAP
- marker gene ranking
- LLM-based cluster annotation with `llama3.1:8b`
- spatial neighborhood/domain analysis
- cell-type colocalization analysis

## Output layout

Under `output_directory`, the pipeline creates:

- `processed/`: processed Zarr data
- `analysis/`: JSON/CSV analysis artifacts
- `figures/`: saved plots
- `logs/`: run logs

## Common failure modes

- If `uv run main.py` fails because `processed.zarr` is missing, run `uv run ingest.py` first.
- If annotation fails, confirm that `ollama serve` is running and that `llama3.1:8b` was downloaded with `ollama pull llama3.1:8b`.
- If paths are wrong, update `config.yaml` before rerunning.
