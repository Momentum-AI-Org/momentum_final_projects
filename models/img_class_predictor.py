import torch.nn as nn
from torchsummary import summary

from utils.constants import IMG_SIZE


class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        depth: int = 6,
        num_filters: int = 64,
    ) -> None:
        super().__init__()

        padding = 1
        conv_kernel_size = 3
        pool_kernel_size = 2

        modules = []
        img_size = IMG_SIZE

        modules.extend(
            [
                nn.Conv2d(
                    3,
                    num_filters,
                    kernel_size=conv_kernel_size,
                    padding=padding,
                ),
            ]
        )

        for _ in range(depth):
            modules.extend(
                [
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=pool_kernel_size),
                    nn.Conv2d(
                        num_filters,
                        num_filters,
                        kernel_size=conv_kernel_size,
                        padding=padding,
                    ),
                ]
            )
            img_size /= pool_kernel_size

        modules.extend(
            [
                nn.Flatten(),
                nn.Linear(int(num_filters * img_size**2), num_classes),
            ]
        )

        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    model = SimpleCNN(
        num_classes=2,
    )
    summary(model, (3, IMG_SIZE, IMG_SIZE))
