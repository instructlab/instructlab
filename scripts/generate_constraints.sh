#!/bin/sh
set -e

# Require Linux; at least until we generate constraints files for each platform
if [ "$(uname)" != "Linux" ]; then
    echo "This script is only supported on Linux."
    exit 1
fi

# If we run from tox, ignore the index url
unset PIP_EXTRA_INDEX_URL

pip-compile \
    --strip-extras \
    --output-file=constraints-dev.txt \
    --constraint constraints-dev.txt.in \
    requirements*.txt docs/requirements.txt
sed '/#.*/d' -i constraints-dev.txt

# TODO: remove after constraint is moved from tox.ini to constraints-dev.txt
sed '/^isort==/d' -i constraints-dev.txt
