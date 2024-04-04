#!/usr/bin/env bash

set -ex

UNAME="$(uname -s)"
if [[ "$UNAME" = "Darwin" ]]; then
    SEDI='sed -i ""'
else
    SEDI='sed -i'
fi

# build a prompt string that includes the time, source file, line number, and function name
export PS4='+$(date +"%Y-%m-%d %T") ${BASH_SOURCE}:${LINENO}: ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'

pip install .

export TEST_CTX_SIZE_LAB_SERVE_LOG_FILE=test_ctx_size_lab_serve.log
export TEST_CTX_SIZE_LAB_CHAT_LOG_FILE=test_ctx_size_lab_chat.log

for cmd in ilab expect; do
    if ! type -p $cmd; then
        echo "Error: $cmd is not installed"
        exit 1
    fi
done

PID_SERVE=
PID_CHAT=

cleanup() {
    set +e
    if [ "${1:-0}" -ne 0 ]; then
        echo "Error occurred in function: ${FUNCNAME[1]} with exit code: $1"
    fi
    for pid in $PID_SERVE $PID_CHAT; do
        if [ -n "$pid" ]; then
            kill $pid
        fi
    done
    rm -f "$TEST_CTX_SIZE_LAB_SERVE_LOG_FILE" \
        "$TEST_CTX_SIZE_LAB_CHAT_LOG_FILE" \
        test_session_history \
        simple_math.yaml
    # revert log level change from test_temp_server()
    $SEDI 's/DEBUG/INFO/g' config.yaml
    # revert port change from test_bind_port()
    $SEDI 's/9999/8000/g' config.yaml
    set -e
}

trap 'cleanup "$?"' EXIT QUIT INT TERM

rm -f config.yaml

# print version
ilab --version

# pipe 3 carriage returns to ilab init to get past the prompts
echo -e "\n\n\n" | ilab init

# Enable Debug in func tests
$SEDI -e 's/log_level:.*/log_level: DEBUG/g;' config.yaml

# download the latest version of the ilab
ilab download

# check that ilab serve is working
test_bind_port(){
    expect -c '
    set timeout 60
    spawn ilab serve
    expect {
        "http://127.0.0.1:8000/docs" { puts OK }
        default { exit 1 }
    }

    # check that ilab serve is detecting the port is already in use
    # catch "error while attempting to bind on address ...."
    spawn ilab serve
    expect {
        "error while attempting to bind on address " { puts OK }
        default { exit 1 }
    }

    # configure a different port
    exec $SEDI 's/8000/9999/g' config.yaml

    # check that ilab serve is working on the new port
    # catch ERROR strings in the output
    spawn ilab serve
    expect {
        "http://127.0.0.1:9999/docs" { puts OK }
        default { exit 1 }
    }
'
}

test_ctx_size(){
    # A context size of 55 will allow a small message
    ilab serve --max-ctx-size 55 &> "$TEST_CTX_SIZE_LAB_SERVE_LOG_FILE" &
    PID_SERVE=$!

    # Make sure the server has time to open the port
    # if "ilab chat" tests it before its open then it will run its own server without --max-ctx-size 1
    sleep 5

    # Should succeed
    ilab chat -qq "Hello"

    # the error is expected so let's ignore it to not fall into the trap
    set +e
    # now chat with the server and exceed the context size
    ilab chat &> "$TEST_CTX_SIZE_LAB_CHAT_LOG_FILE" <<EOF &
hello, I am a ci message that should not finish because I am too long for the context window, tell me about your day please, I love to hear all about it, tell me about the time you could only take 55 tokens
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

test_server_shutdown_while_chatting(){
    # we don't want to fall into the trap function since the failure is expected
    # so we force the command to return 0
    timeout 10 ilab serve || true &

    # add the pid to the list of PID to kill in case something fails before the 5s timeout
    PID_SERVE=$!

    expect -c '
        set timeout 30
        spawn ilab chat
        expect ">>>"
        send "hello! Tell me a long story\r"
        expect {
            "Connection to the server was closed" { exit 0 }
            timeout { exit 1 }
        }
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
    ilab serve --max-ctx-size 128 &
    PID_SERVE=$!

    # chat with the server
    expect -c '
        set timeout 120
        spawn ilab chat
        expect ">>>"
        send "this is a test! what are you? do not exceed ten words in your reply.\r"
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
        spawn ilab chat -s test_session_history
        expect {
            "this is a test! what are you? do not exceed ten words in your reply." { exit 0 }
            timeout { exit 1 }
        }
        send "/l test_session_history\r"
        expect {
            "this is a test! what are you? do not exceed ten words in your reply." { exit 0 }
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
  answer: it is 4
task_description: 'simple maths'
EOF

    $SEDI -e 's/num_instructions:.*/num_instructions: 1/g' config.yaml

    # This should be finished in a minut or so but time it out incase it goes wrong
    timeout 10m ilab generate --taxonomy-path simple_math.yaml

    # Test if generated was created and contains files
    ls -l generated/*
    if [ $(cat $(ls -tr generated/generated_* | tail -n 1) | jq ". | length" ) -lt 1 ] ; then
        echo "Not enough generated results"
        exit 1
    fi
}

test_temp_server(){
    nc -l 8000 --keep-open &
    PID_SERVE=$!
    $SEDI 's/INFO/DEBUG/g' config.yaml
    expect -c '
        set timeout 120
        spawn ilab chat
        expect {
            "Starting a temporary server at" { exit 0 }
            timeout { exit 1 }
        }
    '
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
test_server_shutdown_while_chatting
cleanup
test_temp_server

exit 0
