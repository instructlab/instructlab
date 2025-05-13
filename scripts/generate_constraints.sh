#!/bin/sh
set -xe
pip-compile --output-file=constraints-dev.txt constraints-dev.txt.in requirements*.txt
sed '/#.*/d' -i constraints-dev.txt
sed 's/\[.*\]//' -i constraints-dev.txt

# TODO: remove after constraint is moved from tox.ini to constraints-dev.txt
sed '/^isort==/d' -i constraints-dev.txt
