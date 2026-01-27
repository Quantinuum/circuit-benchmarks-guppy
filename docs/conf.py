# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from inspect import getsourcefile
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path("../src/solarium").absolute()))


project = "solarium"
copyright = "2025, Quantinuum"
author = "Quantinuum"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinxcontrib.plantuml",
    "sphinx_autodoc_typehints",
    "sphinx_design",
]

# Make sure notebooks do not execute while building documentation
nbsphinx_execute = 'never'

# Path to the plantuml.jar file WITHIN the docker image
plantuml = "java -jar /opt/plantuml/plantuml.jar"

# Enable autosummary
autosummary_generate = True
autodoc_member_order = "bysource"

# Allow .md files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

templates_path = ["src/_templates"]
exclude_patterns = [
    "_build", 
    '**/.doctrees',
    '**/.nbsphinx',
    '**/_html',
    '**/generated/*', 
    "Thumbs.db", 
    ".DS_Store"
]

todo_include_todos = True


myst_enable_extensions = [
    "attrs_inline",
    "attrs_block",
    "tasklist",
    "colon_fence",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["src/_static"]
html_css_files = ["custom.css"]

# -- Options for autoapi ----------------------------------------------------
# https://sphinx-autoapi.readthedocs.io/en/latest/tutorials.html
# https://bylr.info/articles/2022/05/10/api-doc-with-sphinx-autoapi/

autoapi_dirs = ["../src/solarium"]
autoapi_type = "python"

# autoapi_options = [
#    "members",
#    "undoc-members",
#    "show-inheritance",
#    "show-module-summary",
#    "imported-members",
# ]

# autoapi_keep_files = True
# autodoc_typehints = "signature"
# autoapi_python_class_content = "both"

html_theme_options = {
    "sidebar_hide_name": True,
}
