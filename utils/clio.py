from subprocess import PIPE, run
from typing import Tuple

from utils.constants import VERBOSITY


def run_command(
    command: str, verbosity: VERBOSITY = VERBOSITY.ERRORS
) -> Tuple[str, str]:
    """Run shell command and get outputs and errors"""
    print(f"> {command}")
    result = run(
        command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True
    )
    if result.stdout and verbosity >= VERBOSITY.DEBUG:
        print(result.stdout)
    if result.stderr and verbosity >= VERBOSITY.ERRORS:
        print(result.stderr)
    return result.stdout, result.stderr
