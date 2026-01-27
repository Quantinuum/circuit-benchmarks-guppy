from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import re
import sys
from typing import Union, Sequence, Optional

from pydantic import BaseModel, Field


# https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#sections
RST_VALID_HEADER_CHARS = frozenset(
    [
        "!",
        '"',
        "#",
        "$",
        "%",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        ":",
        ";",
        "<",
        "=",
        ">",
        "?",
        "@",
        "[",
        "\\",
        "]",
        "^",
        "_",
        "`",
        "{",
        "|",
        "}",
        "~",
    ]
)


class Node(ABC):
    @abstractmethod
    def write(self) -> str: ...

    @abstractmethod
    def format(self, config: dict) -> None: ...


class TocTree(Node):
    def __init__(
        self,
        params: Optional[dict[str, Union[int, float, str]]] = None,
        children: Optional[list[str]] = None,
    ):
        self.params = params if params else dict()
        self.children = children if children else list()

    def write(self) -> str:
        params = [f"   :{k}: {v}" for k, v in self.params.items()]
        params = "\n".join(params)

        children = [f"   {child}" for child in self.children]
        children = "\n".join(children)

        return f".. toctree::\n{params}\n\n{children}\n"

    def format(self, config: Config):
        if config.hide_parents_in_trees:
            self.limit_parents_shown()

    def add_maxdepth(self, maxdepth: int):
        self.params["maxdepth"] = int(self.params["maxdepth"]) + maxdepth

    def limit_parents_shown(self, parents: int = 0):
        """Removes the preceding modules in the header

        Example
            .. toctree::
                :maxdepth: 2

                blinky.cli
                blinky.db_conn

            becomes
            .. toctree::
                :maxdepth: 2

                cli <blinky.cli>
                db_conn <blinky.db_conn>

        """

        children = []
        for child in self.children:
            new_child = ".".join(child.split(".")[-(1 + parents) :])
            children.append(f"{new_child} <{child}>")
        self.children = children


class Header(Node):
    def __init__(self, symbol: str, value: str):
        self.symbol = symbol
        self.value = value

    def write(self) -> str:
        return f"{self.value}\n{self.symbol * len(self.value)}\n"

    def format(self, config: Config):
        self.filter_out_trailing_words(config.filter_header_trailing_words)
        if config.hide_parents_in_trees:
            self.limit_parents_shown()

    def filter_out_trailing_words(self, words: Sequence[str]):
        for word in words:
            if self.value.endswith(word):
                self.value = self.value[0 : len(self.value) - len(word)].strip()

    def limit_parents_shown(self, parents: int = 0):
        """Removes the preceding modules in the header

        Example
            package.submodule.subsubmodule -> subsubmodule
        """

        self.value = ".".join(self.value.split(".")[-(1 + parents) :])


class PyModule(Node):
    def __init__(self, name: str):
        self.name = name

    def write(self) -> str:
        return f".. py:module:: {self.name}\n"

    def format(self, config: Config):
        pass


class AutoModule(Node):
    def __init__(
        self, name: str, params: Optional[dict[str, Union[str, int, float]]] = None
    ):
        self.name = name
        self.params = params if params else dict()

    def write(self) -> str:
        params = [f"   :{k}: {v}" for k, v in self.params.items()]
        params = "\n".join(params)

        return f".. automodule:: {self.name}\n{params}\n"

    def format(self, config: Config):
        pass


def parse_toctree(lines: Sequence[str], index: int) -> tuple[int, TocTree]:
    assert lines[index].startswith(".. toctree::")
    tree = TocTree()

    i = 1
    while index + i < len(lines):
        line = lines[index + i]
        i += 1

        if len(line.strip()) == 0:
            continue

        param_match = re.match(r"^\s+:(?P<name>[^:]+):\s*(?P<value>.*)\s*$", line)
        if param_match:
            tree.params[param_match["name"]] = param_match["value"]
            continue

        link_match = re.match(r"^\s+(?P<link>[\w.]*)\s*$", line)
        if link_match:
            tree.children.append(link_match["link"])
            continue

        return (index + i - 1, tree)

    return (index + i, tree)


def parse_automodule(lines: Sequence[str], index: int) -> tuple[int, AutoModule]:
    assert lines[index].startswith(".. automodule::")
    automod = AutoModule(lines[index].split(":: ")[1])

    i = 1
    while index + i < len(lines):
        line = lines[index + i]
        i += 1

        if len(line.strip()) == 0:
            continue

        param_match = re.match(r"^\s+:(?P<name>[^:]+):\s*(?P<value>.*)\s*$", line)
        if param_match:
            automod.params[param_match["name"]] = param_match["value"]
            continue

        return (index + i, automod)

    return (index + i, automod)


def parse_header(lines: Sequence[str], index: int) -> tuple[int, Header]:
    return (index + 2, Header(lines[index + 1][0], lines[index]))


def parse_py_module(lines: Sequence[str], index: int) -> tuple[int, PyModule]:
    assert lines[index].startswith(".. py:module::")
    return (index + 1, PyModule(lines[index].split(":: ")[1]))


def next_line(lines: Sequence[str], index: int) -> Optional[str]:
    return lines[index + 1] if index + 1 < len(lines) else None


def parse(text: str) -> list[Node]:
    """
    Note, we aren't using the docutils rST parser because it doesn't
    come with an unparser. Therefore we'd have to do as much work and it
    would be much more opaque if we were to use docutils. Slightly
    ridiculous.
    """

    lines = text.splitlines()
    index = 0
    elements = []

    while index < len(lines):
        line = lines[index]
        if len(line.strip()) == 0:
            index += 1
            continue

        if line.startswith(".. py:module::"):
            index, element = parse_py_module(lines, index)
            elements.append(element)
            continue
        elif line.startswith(".. toctree::"):
            index, element = parse_toctree(lines, index)
            elements.append(element)
            continue
        elif line.startswith(".. automodule::"):
            index, element = parse_automodule(lines, index)
            elements.append(element)
            continue
        else:
            next_l = next_line(lines, index)
            if (
                next_l is not None
                and len(next_l.strip()) > 0
                and all(c in RST_VALID_HEADER_CHARS for c in next_l.strip())
            ):
                index, element = parse_header(lines, index)
                elements.append(element)
                continue
            else:
                pass

        raise ValueError(f"Unknown line. Line {index}: '{line}'")

    return elements


def format_single_index_file(text: str, config: Config):
    elements = parse(text)

    formatted_elements = []
    most_recent_header = "None"
    for element in elements:
        element.format(config)

        if isinstance(element, Header):
            if element.value == "Submodules":
                most_recent_header = "Submodules"
                if config.hide_submodule_headers:
                    continue
            elif element.value == "Subpackages":
                most_recent_header = "Subpackages"
                if config.hide_subpackage_headers:
                    continue
            else:
                most_recent_header = "None"

        if isinstance(element, TocTree):
            if most_recent_header == "Subpackages":
                element.add_maxdepth(config.package_depth_increase)

        formatted_elements.append(element)

    return "\n".join(element.write() for element in formatted_elements)


def format_all_index_files(docs_dir, config: Config):
    docs_path = Path(docs_dir)
    rst_files = list(docs_path.glob("*.rst"))
    for rst_file in rst_files:
        text = rst_file.read_text(encoding="utf-8")
        try:
            new_text = format_single_index_file(text, config)
        except Exception as err:
            raise ValueError(f"Unable to parse and format file : {rst_file}") from err

        if new_text != text:
            rst_file.write_text(new_text, encoding="utf-8")
            print(f"Updated {rst_file}")


class Config(BaseModel):
    hide_submodule_headers: bool = False
    hide_subpackage_headers: bool = False
    hide_parents_in_trees: bool = False
    hide_parents_in_headers: bool = False

    package_depth_increase: int = 0
    filter_header_trailing_words: list[str] = Field(default_factory=list)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python format_index_files.py <path_to_rst_dir>")
        sys.exit(1)

    config_json = {
        "hide_submodule_headers": True,
        "hide_subpackage_headers": True,
        "hide_parents_in_trees": True,
        "hide_parents_in_headers": True,
        "package_depth_increase": 1,
        "filter_header_trailing_words": ["namespace", "module"],
    }

    config = Config.model_validate(config_json)

    format_all_index_files(sys.argv[1], config)
