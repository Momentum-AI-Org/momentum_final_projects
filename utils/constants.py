import os
from enum import Enum

# data dirs
RAW_DATA_DIR = os.path.join("data", "raw_data")
PROC_DATA_DIR = os.path.join("data", "proc_data")
CLASS_DATA_INDEX_SUFFIX = "index"

# img processing
IMG_SIZE = 256


class IMG_FORMAT(Enum):
    NUMPY: str = "NUMPY"
    PIL: str = "PIL"


# results
RESULTS_DIR = "results"
EXP_NAME = "v0_basic_model_and_evals"
