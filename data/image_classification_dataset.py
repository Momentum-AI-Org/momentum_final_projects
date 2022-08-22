import json
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from data.data_utils import get_index_file_name, preprocess
from utils.constants import PROC_DATA_DIR


def get_datasets(
    class_names: List[str],
    n_train_per_class: int,
    n_test_per_class: int,
    index_dir=PROC_DATA_DIR,
) -> Tuple[Dataset, Dataset]:
    """Read class indexes, split images into train/test, and return train and test datasets containing the specified number of images."""

    train_imgs = {}
    test_imgs = {}
    for class_name in class_names:

        # read index
        index_path = os.path.join(index_dir, get_index_file_name(class_name))
        assert os.path.exists(
            index_path
        ), f"Cannot find index for class {class_name} at {index_path}"
        read_file = open(index_path)
        class_img_paths = json.load(read_file)

        assert (
            len(class_img_paths) >= n_train_per_class + n_test_per_class
        ), f"Not enough images found for class: {class_name}. Found {len(class_img_paths)} but need {n_train_per_class + n_test_per_class}"

        # train test split
        random_path_order = np.random.choice(
            class_img_paths,
            (n_train_per_class + n_test_per_class,),
            replace=False,
        )

        train_imgs[class_name] = random_path_order[:n_train_per_class]
        test_imgs[class_name] = random_path_order[n_train_per_class:]

    return ImageClassificationDataset(train_imgs), ImageClassificationDataset(
        test_imgs
    )


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        classes_to_paths: Dict,
    ) -> None:
        """Image classification dataset. Takes in a dictionary mapping class names to a list of image paths."""

        super().__init__()

        self.class_names = list(classes_to_paths.keys())
        self.num_classes = len(self.class_names)

        self.class_name_to_index = {}
        self.img_paths = []
        self.class_name_labels = []
        for i, class_name in enumerate(self.class_names):
            self.class_name_to_index[class_name] = i
            for img_path in classes_to_paths[class_name]:
                self.img_paths.append(img_path)
                self.class_name_labels.append(class_name)

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return dataset item at index: img, label, one_hot"""
        img = Image.open(self.img_paths[index])
        img_numpy = preprocess(img)
        class_label = self.class_name_labels[index]
        label = self.class_name_to_index[class_label]
        return img_numpy, class_label, label

    def __len__(self) -> int:
        return len(self.img_paths)


if __name__ == "__main__":
    train_dset, test_dset = get_datasets(
        class_names=["Pizza", "Not Pizza"],
        n_train_per_class=200,
        n_test_per_class=25,
    )

    print(f"Train dataset contains {len(train_dset)} items.")
    print(f"Test dataset contains {len(test_dset)} items.")

    num_print = 10
    for i in np.random.choice(len(train_dset), (num_print,), replace=False):
        _, class_label, label = train_dset[i]
        print(class_label, label)
