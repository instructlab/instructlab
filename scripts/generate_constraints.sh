#!/bin/sh
set -e

# Require Linux; at least until we generate constraints files for each platform
if [ "$(uname)" != "Linux" ]; then
    echo "This script is only supported on Linux."
    exit 1
fi

# If we run from tox, ignore the index url
unset PIP_EXTRA_INDEX_URL

CONSTRAINTS_FILE=constraints-dev.txt

pip-compile \
    --no-header \
    --annotate \
    --annotation-style line \
    --allow-unsafe \
    --strip-extras \
    --output-file=$CONSTRAINTS_FILE \
    --constraint constraints-dev.txt.in \
    requirements*.txt docs/requirements.txt

# clean up empty lines and comments
sed '/^#.*/d' -i constraints-dev.txt
sed '/^$/d' -i constraints-dev.txt

# pip-compile lists -r requirements.txt twice for some reason: once with
# relative path and once with absolute. Clean it up.
sed -E 's/-r \/[^ ]+\/[^,]+, *//' -i $CONSTRAINTS_FILE

# TODO: remove after constraint is moved from tox.ini to constraints-dev.txt
sed '/^isort==/d' -i $CONSTRAINTS_FILE
