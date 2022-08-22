import os
from enum import Enum

# data dirs
RAW_DATA_DIR = os.path.join("data", "raw_data")
PROC_DATA_DIR = os.path.join("data", "proc_data")
TRAIN_SUFFIX = "train"
TEST_SUFFIX = "test"

# img processing
IMG_SIZE = 256


class IMG_FORMAT(Enum):
    NUMPY: str = "NUMPY"
    PIL: str = "PIL"
