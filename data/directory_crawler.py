import os
from typing import List

from utils.constants import RAW_DATA_DIR


def is_file_of_allowed_type(
    file_path: str, allowed_file_types: List[str] = [".png", ".jpg"]
) -> bool:
    """Given the path to a file, determine if the file is of an allowed type"""
    for allowed_file_type in allowed_file_types:
        if file_path.endswith(allowed_file_type):
            return True
    return False


def crawl(
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
            relevant_paths.extend(crawl(child_path, allowed_file_types))
        if os.path.isfile(child_path) and is_file_of_allowed_type(
            child_path, allowed_file_types
        ):
            relevant_paths.append(child_path)

    return relevant_paths


if __name__ == "__main__":
    dir_to_crawl = os.path.join(RAW_DATA_DIR, "archive_recaptcha")
    for file_path in crawl(dir_to_crawl):
        print(file_path)
