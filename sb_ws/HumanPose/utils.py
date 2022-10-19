import os
from pathlib import Path


def convert_abs_path(local_path):
    local_path = str(Path(local_path))  # os-agnostic
    abs_path = os.path.abspath(local_path)  # absolute path
    return abs_path
