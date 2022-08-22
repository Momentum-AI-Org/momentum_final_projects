from api.config import ProjectConfig
from utils.clio import run_command


def setup_script():
    """Set up project environment"""

    assert (
        ProjectConfig.PROJECT_NAME != "default"
    ), "Make sure to set the project name before running this setup script!"
    print(f"Setting up project {ProjectConfig.PROJECT_NAME}...")
    setup_commands = [
        "pip install -r requirements.txt",
    ]
    for command in setup_commands:
        run_command(command)
