import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.constants import (
    PROC_DATA_DIR,
    RAW_DATA_DIR,
    TEST_SUFFIX,
    TRAIN_SUFFIX,
)


def process_raw_imgs(
    query_name: str,
    percent_train: float = 0.8,
):
    """
    Split raw downloaded images into train and test sets.
    """

    print(f"Processing images for query {query_name}:")
    raw_img_dir = os.path.join(RAW_DATA_DIR, query_name)

    assert os.path.exists(
        raw_img_dir
    ), f"No directory exists with images from search query '{query_name}'."

    # randomize
    all_img_names = os.listdir(raw_img_dir)
    random_index_order = np.random.choice(
        len(all_img_names), (len(all_img_names),), replace=False
    )

    assert len(all_img_names) == len(random_index_order)

    train_cutoff = int(len(all_img_names) * percent_train)

    # save train images
    train_imgs_dir = os.path.join(PROC_DATA_DIR, f"{query_name}_{TRAIN_SUFFIX}")
    if not os.path.exists(train_imgs_dir):
        os.makedirs(train_imgs_dir)

        print("Gathering and saving train images...")
        for i, idx in enumerate(tqdm(random_index_order[:train_cutoff])):
            img_name = all_img_names[idx]
            save_img_path = os.path.join(train_imgs_dir, f"{i}.png")
            Image.open(os.path.join(raw_img_dir, img_name)).save(save_img_path)

    # save test images
    test_imgs_dir = os.path.join(PROC_DATA_DIR, f"{query_name}_{TEST_SUFFIX}")
    if not os.path.exists(test_imgs_dir):
        os.makedirs(test_imgs_dir)

        print("Gathering and saving test images...")
        for i, idx in enumerate(tqdm(random_index_order[train_cutoff:])):
            img_name = all_img_names[idx]
            save_img_path = os.path.join(test_imgs_dir, f"{i}.png")
            Image.open(os.path.join(raw_img_dir, img_name)).save(save_img_path)

    print("done.\n")


if __name__ == "__main__":
    process_raw_imgs("apple_fruit")
    process_raw_imgs("orange_fruit")
