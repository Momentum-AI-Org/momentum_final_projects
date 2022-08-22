from api.config import ProjectConfig


def download_data() -> None:
    """Download data needed for projects."""

    assert (
        ProjectConfig.PROJECT_NAME != "default"
    ), "Make sure to set the project name before running this setup script!"

    print(f"Downloading data for project {ProjectConfig.PROJECT_NAME}...")
