import importlib.resources
from pathlib import Path


def data_path(file_name: str) -> Path:
    resources = importlib.resources.files("circuit_benchmarks_guppy")
    resource_path = resources.joinpath("data").joinpath(file_name)
    return resource_path 