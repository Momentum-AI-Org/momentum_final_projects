import os

from api.config import ProjectConfig
from data.dataset_crawler import build_dataset_index
from utils.clio import run_command
from utils.constants import (
    DOWNLOADED_DATASET_ARCHIVE_PATH,
    DOWNLOADED_DATASET_DIR,
    PROJECT_CLASSES,
    PROJECT_DATASET_ARCHIVES,
    PROJECT_TYPE,
    RAW_DATA_DIR,
    VERBOSITY,
)


def download_data() -> None:
    """Download data needed for projects."""

    assert (
        ProjectConfig.PROJECT_NAME != PROJECT_TYPE.DEFAULT
    ), "Make sure to set the project name before running this setup script!"

    print(f"Downloading data for project {ProjectConfig.PROJECT_NAME}...")

    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)

    commands = [
        f"wget {PROJECT_DATASET_ARCHIVES[ProjectConfig.PROJECT_NAME]} -O {DOWNLOADED_DATASET_ARCHIVE_PATH}"
        f"unzip {DOWNLOADED_DATASET_ARCHIVE_PATH} -d {DOWNLOADED_DATASET_DIR}"
    ]

    for command in commands:
        run_command(command, verbosity=VERBOSITY.DEBUG)

    build_dataset_index(
        DOWNLOADED_DATASET_DIR, PROJECT_CLASSES[ProjectConfig.PROJECT_NAME]
    )
