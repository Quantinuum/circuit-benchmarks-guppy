

import importlib
from typing import Literal, overload
from types import ModuleType


@overload
def import_optional(
    name: str, 
    extra: str = "",
    *,
    errors: Literal["warn", "ignore"] = ...,
) -> ModuleType | None:
    ...


@overload
def import_optional(
    name: str, 
    extra: str = "",
    *,
    errors: Literal["raise"] = ...,
) -> ModuleType:
    ...


def import_optional(
    name: str, 
    extra: str = "",
    *,
    errors: Literal["raise", "warn", "ignore"] = "raise",
) -> ModuleType | None:
    """

    Largely copied from pandas:
    https://github.com/pandas-dev/pandas/blob/d3860658b0be74b50ed4aafc0232dd1660139f14/pandas/compat/_optional.py#L107
    """

    assert errors in {"raise", "warn", "ignore"}

    msg = (
        f"`Import {name}` failed. {extra} "
        f"Use your package manager to install the {name} package."
    )
    try:
        module = importlib.import_module(name)
    except ImportError as err:
        if errors == "raise":
            raise ImportError(msg) from err
        return None

    return module


