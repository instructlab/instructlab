#!/bin/sh
set -e

# Require Linux; at least until we generate constraints files for each platform
if [ "$(uname)" != "Linux" ]; then
    echo "This script is only supported on Linux."
    exit 1
fi

pip-compile --output-file=constraints-dev.txt constraints-dev.txt.in requirements*.txt
sed '/#.*/d' -i constraints-dev.txt
sed 's/\[.*\]//' -i constraints-dev.txt

# TODO: remove after constraint is moved from tox.ini to constraints-dev.txt
sed '/^isort==/d' -i constraints-dev.txt
