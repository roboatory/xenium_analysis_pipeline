from __future__ import annotations

from pathlib import Path

from spatialdata_io import xenium
from src import io
from src.config import Configuration

CONFIG_PATH = Path("config.yaml").resolve()

configuration = Configuration()
configuration.load_from_yaml(CONFIG_PATH)
configuration.create_directories()

spatial_data = xenium(configuration.raw_data_directory)
if "morphology_focus" in spatial_data:
    del spatial_data["morphology_focus"]

io.write_spatialdata_zarr(configuration, spatial_data)
