# SPDX-License-Identifier: Apache-2.0

# Standard
import glob
import json

# Third Party
import pytest


# This loads each notebook and attempts to parse the JSON that is contained
# within it in order to validate that it is well-formed JSON. This is intended
# to be a smoke test for notebooks until there is a better defined test story
# for them.
@pytest.mark.parametrize("path", glob.glob("**/*.ipynb"))
def test_notebooks(path):
    with open(path, encoding="utf-8") as notebook_file:
        json.load(notebook_file)
