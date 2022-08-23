import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from data.image_classification_dataset import (
    ImageClassificationDataset,
    get_datasets,
)
from utils.constants import EXP_NAME, RESULTS_DIR, VIS_DSET_FIG_NAME


def visualize_dataset_imgs(
    dset: ImageClassificationDataset,
    num_imgs_per_class: int = 6,
    save_dir: str = os.path.join(RESULTS_DIR, EXP_NAME),
    fig_name: str = VIS_DSET_FIG_NAME,
    show_fig: bool = False,
):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    class_names = dset.class_names
    class_to_index = {}
    for class_name in class_names:
        class_to_index[class_name] = []

    for i, class_name in enumerate(dset.class_name_labels):
        if len(class_to_index[class_name]) < num_imgs_per_class:
            class_to_index[class_name].append(i)

    fig = plt.figure(
        figsize=(
            num_imgs_per_class * 4,
            len(class_names) * 4.5,
        )
    )

    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(len(class_names), num_imgs_per_class),
        axes_pad=0.5,
    )
    grid_iterator = iter(grid)

    for class_name in class_names:
        for idx in class_to_index[class_name]:
            ax = next(grid_iterator)
            img, class_label, _ = dset[idx]
            ax.imshow(img.transpose(1, 2, 0))
            ax.set_title(class_label, fontsize=20)
            ax.axis("off")

    plt.savefig(os.path.join(save_dir, fig_name), bbox_inches="tight")
    if show_fig:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    train_dset, _ = get_datasets(
        class_names=["Pizza", "Not Pizza"],
        n_train_per_class=200,
        n_test_per_class=25,
    )

    visualize_dataset_imgs(train_dset)
