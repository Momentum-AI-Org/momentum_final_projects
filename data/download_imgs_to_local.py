import os

import numpy as np
from bing_image_downloader import downloader
from PIL import Image


def load(path):
    image = Image.open(os.path.join(path)).convert("RGB")
    image = image.resize([256, 256])
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1).reshape([1, 3, 256, 256])
    return arr


def get_imgs(query, num_load, verbose=False):

    imgs_dir = os.path.join(os.path.join("imgs"), query)
    if not os.path.exists(imgs_dir) or len(os.listdir(imgs_dir)) < num_load:
        downloader.download(
            query,
            limit=num_load,
            output_dir=os.path.join("imgs"),
            adult_filter_off=False,
            force_replace=False,
            timeout=1,
            verbose=False,
            filter=".jpg",
        )

    imgs = []
    for img_name in os.listdir(imgs_dir):
        imgs.append(load(os.path.join(imgs_dir, img_name)))
    return imgs


if __name__ == "__main__":
    # get_imgs('apple fruit', 100)
    get_imgs("orange fruit", 100)
