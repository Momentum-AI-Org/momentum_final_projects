import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from data.data_utils import preprocess
from utils.constants import PROC_DATA_DIR, TEST_SUFFIX, TRAIN_SUFFIX


class BingDataset(Dataset):
    def __init__(
        self,
        queries: List[str],
        train: bool,
        n_per_class: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.img_paths = []
        self.class_name_labels = []

        self.num_classes = len(queries)
        self.class_names = queries
        self.class_name_to_index = {}
        for i, class_name in enumerate(queries):
            self.class_name_to_index[class_name] = i
        self.train = train

        for class_name in queries:
            self.load_image_names(class_name, n_per_class)

    def load_image_names(self, class_name: str, n: Optional[int]):
        """Load n images from class name folder into dataset."""

        folder_suffix = TRAIN_SUFFIX if self.train else TEST_SUFFIX
        img_dir = os.path.join(PROC_DATA_DIR, f"{class_name}_{folder_suffix}")

        # TODO: gather images
        assert os.path.exists(
            img_dir
        ), f"Could not find train / test images for query '{class_name}'."

        # read imgs and set length
        img_names = os.listdir(img_dir)
        num_imgs = n if n is not None else len(img_names)
        assert num_imgs <= len(
            img_names
        ), f'Tried to load {n} {"train" if self.train else "test"} \
                images from query {class_name} when only {len(img_names)} exist.'

        # collect path, label, and one hot data
        for img_name in img_names[:num_imgs]:
            self.img_paths.append(os.path.join(img_dir, img_name))
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
    train_dset = BingDataset(
        queries=["apple_fruit", "orange_fruit"],
        train=True,
    )
    test_dset = BingDataset(
        queries=["apple_fruit", "orange_fruit"],
        train=False,
    )

    print(f"Train dataset contains {len(train_dset)} items.")
    print(f"Test dataset contains {len(test_dset)} items.")

    num_print = 10
    for i in np.random.choice(len(train_dset), (num_print,), replace=False):
        _, label, one_hot = train_dset[i]
        print(label, one_hot)
