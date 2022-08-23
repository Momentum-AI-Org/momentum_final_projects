import os
from typing import Tuple

import matplotlib.pyplot as plt
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset

from api.config import ProjectConfig
from data.dataset_crawler import build_dataset_index
from data.image_classification_dataset import get_datasets
from data.visualize_dataset import visualize_dataset_imgs
from evaluate import get_classification_accuracy, visualize_model_predictions
from models.img_class_predictor import SimpleCNN
from train import fit
from utils.clio import run_command
from utils.constants import (
    DOWNLOADED_DATASET_ARCHIVE_PATH,
    DOWNLOADED_DATASET_DIR,
    EXP_NAME,
    PROJECT_CLASSES,
    PROJECT_DATASET_ARCHIVES,
    PROJECT_TYPE,
    RAW_DATA_DIR,
    RESULTS_DIR,
    VERBOSITY,
    VIS_LOSS_CURVES_FIG_NAME,
)


def download_data() -> None:
    """Download data needed for projects."""

    assert (
        ProjectConfig.PROJECT_NAME != PROJECT_TYPE.DEFAULT
    ), "Make sure to set the project name before running this setup script!"

    print(
        f"Downloading data for project {ProjectConfig.PROJECT_NAME}. This may take a few minutes..."
    )

    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)

    commands = [
        f"wget -q {PROJECT_DATASET_ARCHIVES[ProjectConfig.PROJECT_NAME]} -O {DOWNLOADED_DATASET_ARCHIVE_PATH}",
        f"unzip {DOWNLOADED_DATASET_ARCHIVE_PATH} -d {DOWNLOADED_DATASET_DIR}",
    ]

    for command in commands:
        run_command(command, VERBOSITY.ERRORS)

    build_dataset_index(
        DOWNLOADED_DATASET_DIR, PROJECT_CLASSES[ProjectConfig.PROJECT_NAME]
    )


def get_train_test_datasets(
    num_train_imgs_per_class: int,
    num_test_imgs_per_class: int,
) -> Tuple[Dataset, Dataset]:
    """Get train and test datasets with specified sizes."""

    return get_datasets(
        class_names=PROJECT_CLASSES[ProjectConfig.PROJECT_NAME],
        n_train_per_class=num_train_imgs_per_class,
        n_test_per_class=num_test_imgs_per_class,
    )


def get_model(depth: int, num_filters: int):
    """Make a model with the specified depth and number of convolutional filters"""
    return SimpleCNN(
        num_classes=len(PROJECT_CLASSES[ProjectConfig.PROJECT_NAME]),
        depth=depth,
        num_filters=num_filters,
    )


def train_model(
    model: torch.nn.Module,
    train_dset: Dataset,
    test_dset: Dataset,
    epochs: int,
    lr: float,
) -> None:
    """Train the model."""
    fit(model, train_dset, test_dset, epochs, lr)


def display_img(img: PIL.Image.Image, set_size: bool = False):
    if set_size:
        _, ax = plt.subplots(figsize=(12, 12))
    else:
        _, ax = plt.subplots()
    ax.imshow(img)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def visualize_dataset(dataset: Dataset) -> None:
    visualize_dataset_imgs(dataset, show_fig=True)


def evaluate_pretrain_accuracy(
    model: torch.nn.Module,
    test_dataset: Dataset,
) -> float:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return get_classification_accuracy(model, test_dataset, 16, device)


def display_loss_curves() -> None:
    img = Image.open(
        os.path.join(RESULTS_DIR, EXP_NAME, VIS_LOSS_CURVES_FIG_NAME)
    )
    display_img(img, set_size=True)


def evaluate_test_accuracy(
    model: torch.nn.Module,
    test_dataset: Dataset,
) -> float:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return get_classification_accuracy(model, test_dataset, 16, device)


def visualize_predictions(
    model: torch.nn.Module, test_dataset: Dataset
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    visualize_model_predictions(model, test_dataset, 16, device, show_fig=True)
