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

PID_SERVE=
PID_CHAT=

cleanup() {
    set +e
    for pid in $PID_SERVE $PID_CHAT; do
        if [ -n "$pid" ]; then
            kill $pid
        fi
    done
    rm -f "$TEST_CTX_SIZE_LAB_SERVE_LOG_FILE" \
        "$TEST_CTX_SIZE_LAB_CHAT_LOG_FILE" \
        test_session_history \
        simple_math.yaml
    set -e
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
        "http://127.0.0.1:8000/docs" { puts OK }
        default { exit 1 }
    }

    # check that lab serve is detecting the port is already in use
    # catch "error while attempting to bind on address ...."
    spawn lab serve
    expect {
        "error while attempting to bind on address " { puts OK }
        default { exit 1 }
    }

    # configure a different port
    exec sed -i 's/8000/9999/g' config.yaml

    # check that lab serve is working on the new port
    # catch ERROR strings in the output
    spawn lab serve
    expect {
        "http://127.0.0.1:9999/docs" { puts OK }
        default { exit 1 }
    }
'
}

test_ctx_size(){
    lab serve --max-ctx-size 1 &> "$TEST_CTX_SIZE_LAB_SERVE_LOG_FILE" &
    PID_SERVE=$!

    # Make sure the server has time to open the port
    # if "lab chat" tests it before its open then it will run its own server without --max-ctx-size 1
    sleep 5

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

test_loading_session_history(){
    lab serve &
    PID_SERVE=$!

    # chat with the server
    expect -c '
        set timeout 120
        spawn lab chat
        expect ">>>"
        send "hello this is session history test! give me a very short answer!\r"
        send "/s test_session_history\r"
        send "exit\r"
        expect eof
    '

    # verify the session history file was created
    if ! test -f test_session_history; then
        echo "session history file not found"
        exit 1
    fi

    expect -c '
        spawn lab chat -s test_session_history
        expect {
            "hello this is session history test! give me a very short answer!" { exit 0 }
            timeout { exit 1 }
        }
        send "/l test_session_history\r"
        expect {
            "hello this is session history test! give me a very short answer!" { exit 0 }
            timeout { exit 1 }
        }
    '
}

test_generate(){
    cat - <<EOF >  simple_math.yaml
created_by: ci
seed_examples:
- question: what is 1+1
  answer: it is 2
- question: what is 1+3
  answer: 4
task_description: 'simple maths'
EOF

    sed -i -e 's/num_instructions:.*/num_instructions: 1/g' config.yaml

    # This should be finished in a minut or so but time it out incase it goes wrong
    timeout 5m lab generate --taxonomy-path simple_math.yaml

    # Test if generated was created and contains files
    ls -l generated/*
    if [ $(cat $(ls -tr generated/generated_* | tail -n 1) | jq ". | length" ) -lt 1 ] ; then
        echo "Not enough generated results"
        exit 1
    fi
}

########
# MAIN #
########
# call cleanup in-between each test so they can run without conflicting with the server/chat process
test_bind_port
cleanup
test_ctx_size
cleanup
test_loading_session_history
cleanup
test_generate

exit 0
