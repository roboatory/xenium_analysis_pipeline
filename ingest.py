from __future__ import annotations

from pathlib import Path

from spatialdata_io import xenium
from src import io
from src.config import Configuration
from src.logging_utils import get_logger, initialize_logging

CONFIG_PATH = Path("config.yaml").resolve()
logger = get_logger(__name__)

configuration = Configuration()
configuration.load_from_yaml(CONFIG_PATH)
configuration.create_directories()
initialize_logging(configuration.logs_directory, reset=True)
logger.info("ingestion start")

spatial_data = xenium(configuration.raw_data_directory)
if "morphology_focus" in spatial_data:
    del spatial_data["morphology_focus"]
    logger.debug("removed morphology_focus element from ingested spatialdata")

io.write_spatialdata_zarr(configuration, spatial_data)
logger.info("ingestion complete")
