#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
This script validates a Jupyter notebook file.
Usage: python validate_notebook.py <notebook.ipynb>
"""

# Standard
import sys

# Third Party
import nbformat


def validate_notebook(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        nbformat.validate(nb)
        print(f"{file_path} is a valid Jupyter notebook.")
        return True
    except nbformat.ValidationError as e:
        print(f"{file_path} is not a valid Jupyter notebook. Validation error: {e}")
        return False
    except Exception as e:
        print(f"An error occurred while validating {file_path}: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_notebook.py <notebook.ipynb>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    if not validate_notebook(notebook_path):
        sys.exit(1)
