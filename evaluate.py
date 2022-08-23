import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.constants import EXP_NAME, RESULTS_DIR


def get_classification_accuracy(
    model: torch.nn.Module,
    dset: Dataset,
    batch_size: int,
    device: torch.device,
) -> float:

    model.eval()
    dataloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    results = []
    with torch.no_grad():
        for batch_imgs, _, batch_label in tqdm(dataloader):
            batch_imgs = batch_imgs.to(device)
            pred = model(batch_imgs).cpu().numpy()
            pred_class = np.argmax(pred, axis=1)

            for pred_class_index, label in zip(
                pred_class, batch_label.cpu().numpy()
            ):
                if pred_class_index == label:
                    results.append(1)
                else:
                    results.append(0)

    return np.mean(np.array(results))


def visualize_model_predictions(
    model: torch.nn.Module,
    dset: Dataset,
    batch_size: int,
    device: torch.device,
    n_rows: int = 6,
    n_cols: int = 6,
    save_dir: str = os.path.join(RESULTS_DIR, EXP_NAME),
    fig_name: str = "model_predictions.png",
) -> float:

    model.eval()
    dataloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    results = []
    with torch.no_grad():
        for batch_imgs, _, _ in tqdm(dataloader):
            batch_imgs = batch_imgs.to(device)
            pred = model(batch_imgs).cpu().numpy()
            pred_class = np.argmax(pred, axis=1)

            for pred_class_index in pred_class:
                results.append(pred_class_index)

    fig = plt.figure(
        figsize=(
            n_cols * 4,
            n_rows * 4.5,
        )
    )

    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(n_rows, n_cols),
        axes_pad=0.8,
    )
    grid_iterator = iter(grid)

    random_indicies = np.random.choice(
        len(results), (n_rows * n_cols,), replace=False
    )

    for idx in random_indicies:
        ax = next(grid_iterator)
        img, class_label, _ = dset[idx]
        pred_label = dset.class_names[results[idx]]
        ax.imshow(img.transpose(1, 2, 0))
        if class_label == pred_label:
            ax.set_title(
                f"true label: {class_label}\npred label: {pred_label}",
                color="green",
                fontsize=20,
                pad=1.5,
            )
        else:
            ax.set_title(
                f"true label: {class_label}\npred label: {pred_label}",
                color="red",
                fontsize=20,
                pad=1.5,
            )
        ax.axis("off")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, fig_name), bbox_inches="tight")
    plt.close()


def visualize_loss_curves(
    train_loss: List[float],
    val_loss: List[float],
    save_dir: str = os.path.join(RESULTS_DIR, EXP_NAME),
    fig_name: str = "loss_curves.png",
) -> float:

    _, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color=color)
    ax1.plot(
        [i for i in range(len(train_loss))],
        train_loss,
        color=color,
        label="training loss",
    )
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("accuracy (%)", color=color)
    ax2.plot(
        [i for i in range(len(val_loss))],
        val_loss,
        color=color,
        label="validation accuracy",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Training Loss and Validation Accuracy Over Training")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, fig_name), bbox_inches="tight")
    plt.close()
