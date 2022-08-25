import os
from enum import Enum, IntEnum

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

VIS_DSET_FIG_NAME = "dataset_by_class.png"
VIS_MODEL_PREDS_FIG_NAME = "model_predictions.png"
VIS_LOSS_CURVES_FIG_NAME = "loss_curves.png"


# running cl commands
class VERBOSITY(IntEnum):
    NONE = 0
    ERRORS = 1
    WARNINGS = 2
    DEBUG = 3


# api
class PROJECT_TYPE(Enum):
    DEFAULT = "DEFAULT"
    SHOES = "SHOES"
    FRUIT = "FRUIT"
    PIZZA = "PIZZA"
    RECAPTCHA = "RECAPTCHA"
    WEATHER = "WEATHER"
    MICROORGANISM = "MICROORGANISM"
    DEVDIGIT = "DEVDIGIT"
    AUTOMOBILE = "AUTOMOBILE"
    ANIMALS = "ANIMALS"


DOWNLOADED_DATASET_ARCHIVE_PATH = os.path.join(RAW_DATA_DIR, "archive.zip")
DOWNLOADED_DATASET_DIR = os.path.join(RAW_DATA_DIR, "unziped_dataset")

PROJECT_DATASET_ARCHIVES = {
    PROJECT_TYPE.SHOES: "https://www.dropbox.com/s/7gtcc0yv7zqcr2j/archive_shoes.zip",
    PROJECT_TYPE.FRUIT: "https://www.dropbox.com/s/are7px0flw44vn0/archive_fruit.zip",
    PROJECT_TYPE.PIZZA: "https://www.dropbox.com/s/t68s1vz5r3uzu16/archive_pizza.zip",
    PROJECT_TYPE.RECAPTCHA: "https://www.dropbox.com/s/s0873h0xa5318qj/archive_recaptcha.zip",
    PROJECT_TYPE.WEATHER: "https://www.dropbox.com/s/sif4kv7oxc2b5lq/archive_weather.zip",
    PROJECT_TYPE.MICROORGANISM: "https://www.dropbox.com/s/o334ofu27xq7boe/microOrganism.zip",
    PROJECT_TYPE.DEVDIGIT: "https://www.dropbox.com/s/29j9otrmgho37ap/devDigits.zip",
    PROJECT_TYPE.AUTOMOBILE: "https://www.dropbox.com/s/uzddilfpx8b0l90/automobile.zip",
    PROJECT_TYPE.ANIMALS: "https://www.dropbox.com/s/zv9246of90flk1t/archive.zip?dl=0",
}

PROJECT_CLASSES = {
    PROJECT_TYPE.SHOES: ["Adidas", "Converse", "Nike"],
    PROJECT_TYPE.FRUIT: ["Apple", "Banana", "Orange"],
    PROJECT_TYPE.PIZZA: ["Pizza", "Not Pizza"],
    PROJECT_TYPE.RECAPTCHA: [
        "Bicycle",
        "Bridge",
        "Bus",
        "Car",
        # "Chimney",
        "Crosswalk",
        "Hydrant",
        # "Motorcycle",
        # "Other",
        "Palm",
        # "Stair",
        "Traffic Light",
    ],
    PROJECT_TYPE.WEATHER: [
        "Dew",
        "Fogsmog",
        "Frost",
        "Glaze",
        "Hail",
        "Lightning",
        # "Rain",
        # "Rainbow",
        "Rime",
        "Sandstorm",
        "Snow",
    ],
    PROJECT_TYPE.MICROORGANISM: [
        "Amoeba",
        "Euglena",
        "Hydra",
        "Paramecium",
        "Rod_bacteria",
        "Spherical_bacteria",
        "Spiral_bacteria",
        "Yeast",
    ],
    PROJECT_TYPE.DEVDIGIT: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    PROJECT_TYPE.AUTOMOBILE: [
        "Bevel-gear",
        "bearing",
        "clutch",
        "cylincer",
        "filter",
        "fuel-tank",
        "helical_gear",
        "piston",
        "rack-pinion",
        "shocker",
        "spark-plug",
        "spur-gear",
        "wheel",  # , "valve"
    PROJECT_TYPE.ANIMALS: [
        "butterfly",
        "chicken",
        "dog",
        "horse",
        "spider",
        "cat",
        "cow",
        "elephant",
        "sheep",
        "squirrel",
    ],
}
