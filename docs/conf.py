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
    "enum_tools.autoenum",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_click",
    "sphinxcontrib.autodoc_pydantic",
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
nitpick_ignore_regex = [
    # instructlab.configuration
    ("py:class", "annotated_types\..*"),
    ("py:class", "pydantic\.types\..*"),
    ("py:obj", "typing\..*"),
    ("py:obj", "instructlab\.configuration\..*"),
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["static"]

# autodoc settings
# add_module_names = False

# https://autodoc-pydantic.readthedocs.io/
# settings
autodoc_pydantic_model_show_json = False
autodoc_pydantic_settings_show_json = False
autodoc_pydantic_model_show_config_summary = False
# fields
autodoc_pydantic_model_show_field_summary = True
# validators
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
