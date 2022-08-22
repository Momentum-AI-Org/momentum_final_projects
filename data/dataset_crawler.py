import json
import os
import re
from typing import List

from data.data_utils import get_index_file_name
from utils.constants import PROC_DATA_DIR, RAW_DATA_DIR


def is_file_of_allowed_type(
    file_path: str, allowed_file_types: List[str] = [".png", ".jpg"]
) -> bool:
    """Given the path to a file, determine if the file is of an allowed type"""
    for allowed_file_type in allowed_file_types:
        if file_path.endswith(allowed_file_type):
            return True
    return False


def crawl_directory(
    dir: str, allowed_file_types: List[str] = [".png", ".jpg"]
) -> List[str]:
    """Crawl a directory and return all files of given types"""

    assert os.path.exists(
        dir
    ), f"Cannot crawl directory {dir} because directory does not exist!"

    children = os.listdir(dir)
    relevant_paths = []
    for child in children:
        child_path = os.path.join(dir, child)
        if os.path.isdir(child_path):
            relevant_paths.extend(
                crawl_directory(child_path, allowed_file_types)
            )
        if os.path.isfile(child_path) and is_file_of_allowed_type(
            child_path, allowed_file_types
        ):
            relevant_paths.append(child_path)

    return relevant_paths


def tokenize(
    x: str, separators: str = r" |/|\\", strip_characters: str = [" ", "_"]
) -> str:
    """Tokenize a string by splitting at separators"""

    cleaned_tokens = []
    for token in re.split(separators, x):
        if len(token) == 0:
            continue
        for strip_character in strip_characters:
            token = token.replace(strip_character, "")
        cleaned_tokens.append(token)

    return cleaned_tokens


def score_path_query_relevancy(
    path: str,
    query: str,
) -> float:
    """Score how likely a query (class name) is located at the given path based on naming similarities"""
    path_tokens = tokenize(path.lower())
    query_tokens = tokenize(query.lower())

    big_query_token = ""
    for token in query_tokens:
        big_query_token += token
    query_tokens.append(big_query_token)

    score = 0

    for path_token in path_tokens:
        if len(path_token) == 0:
            continue
        for query_token in query_tokens:
            if len(query_token) == 0:
                continue
            if query_token == path_token:
                score += 5
            elif query_token in path_token:
                score += 0.5

    return score


def build_dataset_index(
    dataset_dir: str,
    classes_to_find: List[str],
    index_write_dir: str = PROC_DATA_DIR,
) -> None:
    """Read raw dataset data from directory, discover images of the relevant classes,
    and save data in index file."""

    print(
        f"Crawling {dataset_dir} and attempting to discover image classes: {classes_to_find}..."
    )

    image_paths = crawl_directory(dataset_dir)
    class_paths = {}
    for class_name in classes_to_find:
        class_paths[class_name] = []

    for path in image_paths:
        best_class, best_score = None, 0
        for class_name in classes_to_find:
            score = score_path_query_relevancy(path, class_name)
            if score > best_score:
                best_score = score
                best_class = class_name

        if best_class is None:
            continue
        class_paths[best_class].append(path)

    for key, value in class_paths.items():
        print(f"Class {key} ... found {len(value)} images")
    print("\n")

    if not os.path.exists(index_write_dir):
        os.makedirs(index_write_dir)
    for key, value in class_paths.items():
        index_path = os.path.join(index_write_dir, get_index_file_name(key))
        with open(index_path, "w") as out_file:
            json.dump(value, out_file)


if __name__ == "__main__":
    dir_to_crawl = os.path.join(RAW_DATA_DIR, "archive_recaptcha")
    build_dataset_index(
        dataset_dir=dir_to_crawl,
        classes_to_find=[
            "Bicycle",
            "Bridge",
            "Bus",
            "Car",
            "Chimney",
            "Crosswalk",
            "Hydrant",
            "Motorcycle",
            "Other",
            "Palm",
            "Stair",
            "Traffic Light",
        ],
    )

    dir_to_crawl = os.path.join(RAW_DATA_DIR, "archive_pizza")
    build_dataset_index(
        dataset_dir=dir_to_crawl, classes_to_find=["Pizza", "Not Pizza"]
    )
