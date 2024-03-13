#!/usr/bin/env bash

# SPDX-FileCopyrightText: The InstructLab Authors
# SPDX-License-Identifier: Apache-2.0

set -ex

pip install .

for cmd in lab expect; do
    if ! type -p $cmd; then
        echo "Error: $cmd is not installed"
        exit 1
    fi
done

PID1=
PID2=

cleanup() {
    set +e
    if [ -n "$PID1" ]; then
        kill $PID1
    fi
    if [ -n "$PID2" ]; then
        kill $PID2
    fi
}

trap cleanup 0

rm -f config.yaml

# pipe 3 carriage returns to lab init to get past the prompts
echo -e "\n\n\n" | lab init

# download the latest version of the lab
lab download

# check that lab serve is working
expect -c '
spawn lab serve
expect {
    "http://localhost:8000/docs" { exit 0 }
    eof
}

python -m http.server 8000 &
PID1=$!

# check that lab serve is detecting the port is already in use
# catch "error while attempting to bind on address ...."
spawn lab serve
expect {
    "error while attempting to bind on address " { exit 1 }
    eof
}

# configure a different port
sed -i 's/8000/9999/g' config.yaml

# check that lab serve is working on the new port
# catch ERROR strings in the output
spawn lab serve
expect {
    "http://localhost:9999/docs" { exit 0}
    eof
}
'

python -m http.server 9999 &
PID2=$!

exit 0
