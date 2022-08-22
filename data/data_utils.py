from typing import Callable, List, Union

import numpy as np
import PIL
from PIL import Image

from utils.constants import CLASS_DATA_INDEX_SUFFIX, IMG_FORMAT, IMG_SIZE


def get_index_file_name(x: str, suffix: str = CLASS_DATA_INDEX_SUFFIX) -> str:
    """Get name of index file storing class image data from name of class"""
    x = x.lower().replace(" ", "_")
    return f"{x}_{CLASS_DATA_INDEX_SUFFIX}.json"


def model_ready_img(img: PIL.Image.Image) -> PIL.Image.Image:
    """Return a model ready PIL image."""
    return img.convert("RGB").resize(
        [IMG_SIZE, IMG_SIZE], resample=Image.BICUBIC
    )


def img_to_numpy(img: PIL.Image.Image) -> PIL.Image.Image:
    """Send a PIL image to numpy."""
    return np.array(img).astype(np.float32).transpose(2, 0, 1) / 255.0


def preprocess(
    x: Union[List[PIL.Image.Image], PIL.Image.Image],
    img_format: IMG_FORMAT = IMG_FORMAT.NUMPY,
    to_model_ready: Callable[
        [PIL.Image.Image], PIL.Image.Image
    ] = model_ready_img,
    img_to_numpy: Callable[[PIL.Image.Image], np.ndarray] = img_to_numpy,
) -> Union[
    np.ndarray, PIL.Image.Image, List[np.ndarray], List[PIL.Image.Image]
]:

    if isinstance(x, list):
        out = []

        # model ready
        for img in x:
            out.append(to_model_ready(img))
        if img_format == IMG_FORMAT.PIL:
            return out

        # to numpy
        for i in range(out):
            out[i] = img_to_numpy(out[i])
        return out

    out = to_model_ready(x)
    if img_format == IMG_FORMAT.PIL:
        return out
    return img_to_numpy(out)
