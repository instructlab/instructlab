#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -ex

# build a prompt string that includes the time, source file, line number, and function name
export PS4='+$(date +"%Y-%m-%d %T") ${BASH_VERSION}:${BASH_SOURCE}:${LINENO}: ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'

pip install .

export TEST_CTX_SIZE_LAB_SERVE_LOG_FILE=test_ctx_size_lab_serve.log
export TEST_CTX_SIZE_LAB_CHAT_LOG_FILE=test_ctx_size_lab_chat.log

for cmd in ilab expect timeout; do
    if ! type -p $cmd; then
        echo "Error: $cmd is not installed"
        exit 1
    fi
done

PID_SERVE=
PID_CHAT=

chat_shot="ilab chat -qq"

cleanup() {
    set +e
    if [ "${1:-0}" -ne 0 ]; then
        echo "Error occurred in function: ${FUNCNAME[1]} with exit code: $1"
    fi
    for pid in $PID_SERVE $PID_CHAT; do
        if [ -n "$pid" ]; then
            kill "$pid"
            wait "$pid"
        fi
    done
    rm -f "$TEST_CTX_SIZE_LAB_SERVE_LOG_FILE" \
        "$TEST_CTX_SIZE_LAB_CHAT_LOG_FILE" \
        test_session_history
    rm -rf test_taxonomy
    # revert port change from test_bind_port()
    sed -i.bak 's/9999/8000/g' config.yaml
    # revert model name change from test_model_print()
    sed -i.bak "s/baz/merlinite-7b-lab-Q4_K_M/g" config.yaml
    mv models/foo.gguf models/merlinite-7b-lab-Q4_K_M.gguf || true
    rm -f config.yaml.bak serve.log
    set -e
}

trap 'cleanup "$?"' EXIT QUIT INT TERM

rm -f config.yaml

# print version
ilab --version

# print system information
ilab sysinfo

# pipe 3 carriage returns to ilab init to get past the prompts
echo -e "\n\n\n" | ilab init

# Enable Debug in func tests
sed -i.bak -e 's/log_level:.*/log_level: DEBUG/g;' config.yaml

# It looks like GitHub action MacOS runner does not have graphics
# so we need to disable the GPU layers if we are running in GitHub actions
if [[ "$(uname)" == "Darwin" ]]; then
    if command -v system_profiler; then
        if system_profiler SPDisplaysDataType|grep "Metal Support"; then
            echo "Metal GPU detected"
        else
            echo "No Metal GPU detected"
            sed -i.bak -e 's/gpu_layers: -1/gpu_layers: 0/g;' config.yaml
        fi
    else
        echo "system_profiler not found, cannot determine GPU"
    fi
fi

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
    exec sed -i.bak 's/8000/9999/g' config.yaml

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
    ilab serve --max-ctx-size 25 &> "$TEST_CTX_SIZE_LAB_SERVE_LOG_FILE" &
    PID_SERVE=$!

    # Make sure the server has time to open the port
    # if "ilab chat" tests it before its open then it will run its own server without --max-ctx-size 1
    wait_for_server

    # SHOULD SUCCEED: ilab chat will trim the SYS_PROMPT then take the second message
    ${chat_shot} "Hello"

    # SHOULD FAIL: ilab chat will trim the SYS_PROMPT AND the second message, then raise an error
    # The errors from failures will be written into the serve log and chat log files
    ${chat_shot} "hello, I am a ci message that should not finish because I am too long for the context window, tell me about your day please?
    How many tokens could you take today. Could you tell me about the time you could only take twenty five tokens" &> "$TEST_CTX_SIZE_LAB_CHAT_LOG_FILE" &
    PID_CHAT=$!

    # look for the context size error in the server logs
    if ! timeout 20 bash -c '
        until grep -q "exceed context window of" "$TEST_CTX_SIZE_LAB_SERVE_LOG_FILE"; do
        echo "waiting for context size error"
        sleep 1
    done
'; then
        echo "context size error not found in server logs"
        cat $TEST_CTX_SIZE_LAB_SERVE_LOG_FILE
        exit 1
    fi

    # look for the context size error in the chat logs
    if ! timeout 20 bash -c '
        until grep -q "Message too large for context size." "$TEST_CTX_SIZE_LAB_CHAT_LOG_FILE"; do
        echo "waiting for chat error"
        sleep 1
    done
'; then
        echo "context size error not found in chat logs"
        cat $TEST_CTX_SIZE_LAB_CHAT_LOG_FILE
        exit 1
    fi
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
    mkdir -p test_taxonomy/compositional_skills
    cat - <<EOF >  test_taxonomy/compositional_skills/simple_math.yaml
created_by: ci
version: 2
seed_examples:
  - question: what is 1+1
    answer: it is 2
  - question: what is 1+3
    answer: it is 4
  - question: what is 2+3
    answer: it is 5
  - question: what is 2+5
    answer: it is 7
  - question: what is 7+3
    answer: it is 10
task_description: "simple maths"
EOF

    sed -i.bak -e 's/num_instructions:.*/num_instructions: 1/g' config.yaml

    # This should be finished in a minut or so but time it out incase it goes wrong
    timeout 10m ilab generate --taxonomy-path test_taxonomy/compositional_skills/simple_math.yaml

    # Test if generated was created and contains files
    ls -l generated/*
    if [ $(cat $(ls -tr generated/generated_* | tail -n 1) | jq ". | length" ) -lt 1 ] ; then
        echo "Not enough generated results"
        exit 1
    fi
}

test_temp_server(){
    expect -c '
        set timeout 120
        spawn ilab chat
        expect {
            "Starting a temporary server at" { exit 0 }
            timeout { exit 1 }
        }
    '
}

test_temp_server_sigint(){
    expect -c '
        set timeout 30
        spawn ilab chat
        expect "Starting a temporary server at"
        send "hello!\r"
        sleep 5
        send "\003"
        expect "Disconnected from client (via refresh/close)"
        send "\003"
        send "hello again\r"
        sleep 1
        expect {
            "Connection error, try again" { exit 1 }
        }
    '
}

test_no_chat_logs(){
    expect -c '
        set timeout 30
        spawn ilab chat
        expect "Starting a temporary server at"
        send "hello!\r"
        sleep 1
        expect {
            "_base_client.py" { exit 1 }
        }
        '
}

test_temp_server_ignore_internal_messages(){
    expect -c '
        set timeout 20
        spawn ilab chat
        expect "Starting a temporary server at"
        send "hello!\r"
        sleep 5
        send "\003"
        expect {
            "Disconnected from client (via refresh/close)" { exit 1 }
        }
        send "exit\r"
        expect {
            "Traceback (most recent call last):" { set exp_result 1 }
            default { set exp_result 0 }
        }
        if { $exp_result != 0 } {
            puts stderr "Error: ilab chat command failed"
            exit 1
        }
    '
}

test_server_welcome_message(){
    # test that the server welcome message is displayed
    ilab serve --log-file serve.log &
    PID_SERVE=$!

    wait_for_server

    if ! timeout 10 bash -c '
        until test -s serve.log; do
        echo "waiting for server log file to be created"
        sleep 1
    done
    '; then
        echo "server log file was not created"
        exit 1
    fi
}

wait_for_server(){
    if ! timeout 30 bash -c '
        until curl 127.0.0.1:8000|grep "{\"message\":\"Hello from InstructLab! Visit us at https://instructlab.ai\"}"; do
            echo "waiting for server to start"
            sleep 1
        done
    '; then
        echo "server did not start"
        exit 1
    fi
}

test_model_print(){
    mv models/merlinite-7b-lab-Q4_K_M.gguf models/foo.gguf
    ilab serve --model-path models/foo.gguf &
    PID_SERVE=$!

    wait_for_server

    # validate that we print the model from the server since it is different from the config
    expect -c '
        spawn ilab chat
        expect {
            -re "Welcome to InstructLab Chat w/ \\\u001b\\\[1mMODELS/FOO\\.GGUF" { exit 0 }
            eof { catch wait result; exit [lindex $result 3] }
            timeout { exit 1 }
        }
    '

    # validate that we fail on invalid model
    expect -c '
        set timeout 30
        spawn ilab chat -m bar
        expect {
           "Executing chat failed with: Model bar is not served by the server. These are the served models: ['models/foo.gguf']" { exit 0 }
        }
    '

    # If we don't specify a model, validate that we print the model reported
    # by the server since it is different from the config.
    expect -c '
        spawn ilab chat
        expect {
            -re "Welcome to InstructLab Chat w/ \\\u001b\\\[1mMODELS/FOO\\.GGUF" { exit 0 }
            eof { catch wait result; exit [lindex $result 3] }
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
cleanup
test_temp_server_sigint
cleanup
test_no_chat_logs
cleanup
test_temp_server_ignore_internal_messages
cleanup
test_server_welcome_message
cleanup
test_model_print

exit 0
