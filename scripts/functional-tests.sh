#!/usr/bin/env bash

set -ex

pip install .

export TEST_CTX_SIZE_LAB_SERVE_LOG_FILE=test_ctx_size_lab_serve.log
export TEST_CTX_SIZE_LAB_CHAT_LOG_FILE=test_ctx_size_lab_chat.log

for cmd in lab expect; do
    if ! type -p $cmd; then
        echo "Error: $cmd is not installed"
        exit 1
    fi
done

PID1=
PID2=
PID_SERVE=
PID_CHAT=

cleanup() {
    set +e
    for pid in $PID1 $PID2 $PID_SERVE $PID_CHAT; do
        if [ -n "$pid" ]; then
            kill $pid
        fi
    done
    rm -f "$TEST_CTX_SIZE_LAB_SERVE_LOG_FILE" \
        "$TEST_CTX_SIZE_LAB_CHAT_LOG_FILE"
}

trap cleanup EXIT QUIT INT TERM

rm -f config.yaml

# pipe 3 carriage returns to lab init to get past the prompts
echo -e "\n\n\n" | lab init

# download the latest version of the lab
lab download

# check that lab serve is working
test_bind_port(){
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
}

test_ctx_size(){
    lab serve --max-ctx-size 1 &> "$TEST_CTX_SIZE_LAB_SERVE_LOG_FILE" &
    PID_SERVE=$!

    # the error is expected so let's ignore it to not fall into the trap
    set +e
    # now chat with the server
    lab chat &> "$TEST_CTX_SIZE_LAB_CHAT_LOG_FILE" <<EOF &
hello
EOF
    PID_CHAT=$!
    wait_for_pid_to_disappear $PID_CHAT
    # reset the PID_CHAT variable so that the cleanup function doesn't try to kill it
    PID_CHAT=

    # re-activate the error trap
    set -e

    # look for the context size error in the server logs
    timeout 10 bash -c '
        until grep -q "exceed context window of" "$TEST_CTX_SIZE_LAB_SERVE_LOG_FILE"; do
        echo "waiting for context size error"
        sleep 1
    done
'

    # look for the context size error in the chat logs
    timeout 10 bash -c '
        until grep -q "Message too large for context size." "$TEST_CTX_SIZE_LAB_CHAT_LOG_FILE"; do
        echo "waiting for chat error"
        sleep 1
    done
'
}

wait_for_pid_to_disappear(){
    for i in $(seq 1 20); do
        if ! test -d /proc/$1; then
            break
        fi
        # error if the process is still running
        if [ $i -eq 20 ]; then
            echo "chat process is still running"
            exit 1
        fi
        sleep 1
    done
}

########
# MAIN #
########
# call cleanup in-between each test so they can run without conflicting with the server/chat process
test_bind_port
cleanup
test_ctx_size
cleanup

exit 0
