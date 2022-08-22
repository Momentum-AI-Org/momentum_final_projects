from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tqdm import tqdm

from data.bing_dataset import BingDataset
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

    model.eval()
    dataloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
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


def main(
    queries: List[str],
    epochs: int,
    lr: float,
    eval_batch_size: int = 16,
):

    """Main training script."""

    train_dset = BingDataset(
        queries=queries,
        train=True,
    )
    eval_dset = BingDataset(
        queries=queries,
        train=False,
    )

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

    model = SimpleCNN(
        num_classes=len(queries),
    )
    summary(model, (3, IMG_SIZE, IMG_SIZE))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    # training loop
    for epoch in range(epochs):
        train_losses = []

        print(
            round_with_prec(
                val_step(model, eval_dset, eval_batch_size, device), prec=4
            )
        )

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

    print(
        round_with_prec(
            val_step(model, eval_dset, eval_batch_size, device), prec=4
        )
    )


if __name__ == "__main__":
    main(
        queries=["apple_fruit", "orange_fruit"],
        epochs=30,
        lr=1e-3,
    )
