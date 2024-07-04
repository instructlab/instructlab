# SPDX-License-Identifier: Apache-2.0
#
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "InstructLab"
copyright = "2024, InstructLab Authors"
author = "InstructLab Authors"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_click",
]

templates_path = ["templates"]
exclude_patterns = [
    "build",
    "Thumbs.db",
    ".DS_Store",
    "cli_reference.md",  # ignored in favor of auto-generated docs
]

intersphinx_mapping = {
    # "python": ("https://docs.python.org/", None),
    # "torch": ("https://pytorch.org/docs/stable/", None),
    # "click": ("https://click.palletsprojects.com/en/latest/", None),
    # "pydantic": ("https://docs.pydantic.dev/latest/", None),
    # "transformers": ("https://huggingface.co/docs/transformers/main/en", None),
}

autodoc_mock_imports = [
    "fire",
    "mlx",
]

nitpick_ignore = [
    ("py:class", "git.repo.base.Repo"),
    ("py:class", "uvicorn.config.Config"),
    ("py:class", "uvicorn.server.Server"),
    ("py:class", "transformers.generation.stopping_criteria.StoppingCriteria"),
    ("py:class", "mlx.core.array"),
    ("py:class", "mlx.nn.LayerNorm"),
    ("py:class", "mlx.nn.Linear"),
    ("py:class", "mlx.nn.Module"),
    ("py:class", "nn.Module"),
    ("py:class", "mx.array"),
    # stdlib
    ("py:class", "FrameType"),
    # pydantic auto-generated doc strings
    ("py:class", "ComputedFieldInfo"),
    ("py:class", "ConfigDict"),
    ("py:class", "FieldInfo"),
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["static"]
