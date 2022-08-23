import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tqdm import tqdm

from data.image_classification_dataset import get_datasets
from evaluate import (
    get_classification_accuracy,
    visualize_loss_curves,
    visualize_model_predictions,
)
from models.img_class_predictor import SimpleCNN
from utils.constants import IMG_SIZE
from utils.helpers import round_with_prec


def train_step(
    batch_x: np.ndarray,
    batch_y: np.ndarray,
    model: torch.nn.Module,
    loss_func,
    optimizer,
) -> float:
    """Training step. Returns loss"""

    model.train()
    optimizer.zero_grad()
    pred = model(batch_x)
    loss = loss_func(pred, batch_y)
    loss.backward()
    optimizer.step()

    return loss.item()


def val_step(
    model: torch.nn.Module,
    dset: Dataset,
    batch_size: int,
    device: torch.device,
) -> float:
    """Validation step. Return classification accuracy"""
    return get_classification_accuracy(model, dset, batch_size, device)


def fit(
    model: torch.nn.Module,
    train_dset: Dataset,
    eval_dset: Dataset,
    epochs: int,
    lr: float,
    eval_batch_size: int = 16,
):
    """Main training script."""

    train_dataloader = DataLoader(
        train_dset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    # training loop
    batch_train_losses = []
    batch_validate_losses = []
    for epoch in range(epochs):
        train_losses = []

        for batch_imgs, _, batch_one_hot in (
            pbar_batch := tqdm(train_dataloader)
        ):

            batch_imgs = batch_imgs.to(device)
            batch_one_hot = batch_one_hot.to(device)
            train_losses.append(
                train_step(
                    batch_imgs, batch_one_hot, model, loss_func, optimizer
                )
            )

            avg_train_loss = np.mean(np.array(train_losses))
            pbar_batch.set_description(
                f"Epoch {epoch}/{epochs} | Train Loss: {round_with_prec(avg_train_loss, 4)}"
            )

        avg_train_loss = np.mean(np.array(train_losses))
        val_loss = round_with_prec(
            val_step(model, eval_dset, eval_batch_size, device), prec=4
        )
        print(f"Validation loss: {val_loss}")

        batch_train_losses.append(avg_train_loss)
        batch_validate_losses.append(val_loss)
        visualize_loss_curves(batch_train_losses, batch_validate_losses)

    visualize_model_predictions(model, eval_dset, eval_batch_size, device)


if __name__ == "__main__":
    class_names = ["Pizza", "Not Pizza"]
    train_dset, eval_dset = get_datasets(
        class_names=class_names,
        n_train_per_class=200,
        n_test_per_class=25,
    )
    model = SimpleCNN(
        num_classes=len(class_names),
    )
    summary(model, (3, IMG_SIZE, IMG_SIZE))

    fit(
        model=model,
        train_dset=train_dset,
        eval_dset=eval_dset,
        epochs=30,
        lr=1e-3,
    )
