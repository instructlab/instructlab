#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# This test script is laid out as follows:
# - UTILITIES:  utility functions
# - TESTS:      test functions
# - SETUP:      environment setup steps
# - MAIN:       test execution steps
#
# If you are running locally and calling the script multiple times you may want to run like this:
#
# TEST_DIR=/tmp/foo ./scripts/functional-tests.sh
#
# As soon as TEST_DIR is set to any non-empty value, the test directory will NOT be removed
# This assumes you control your test directory TEST_DIR and its cleanup

set -ex

#############
# UTILITIES #
#############
init_config() {
    # Generate initial configuration file
    ilab config init --non-interactive

    # It looks like GitHub action MacOS runner does not have graphics
    # so we need to disable the GPU layers if we are running in GitHub actions
    if [[ "$(uname)" == "Darwin" ]]; then
        if command -v system_profiler; then
            if system_profiler SPDisplaysDataType|grep "Metal Support"; then
                echo "Metal GPU detected"
            else
                echo "No Metal GPU detected"
                sed -i.bak -e 's/gpu_layers: -1/gpu_layers: 0/g;' "${ILAB_CONFIG_FILE}"
            fi
        else
            echo "system_profiler not found, cannot determine GPU"
        fi
    fi

    # Enable Debug in func tests with debug level 1
    sed -i.bak -e 's/log_level:.*/log_level: DEBUG/g;' "${ILAB_CONFIG_FILE}"
    sed -i.bak -e 's/debug_level:.*/debug_level: 1/g;' "${ILAB_CONFIG_FILE}"
}

cleanup() {
    set +e
    if [ "${1:-0}" -ne 0 ]; then
        printf "\n\n"
        printf "\e[31m=%.0s\e[0m" {1..80}
        printf "\n\e[31mERROR OCCURRED\e[0m\n"
        printf "\e[31mFunction: %s\e[0m\n" "${FUNCNAME[1]}"
        printf "\e[31mExit Code: %s\e[0m\n" "$1"
        printf "\e[31m=%.0s\e[0m" {1..80}
        printf "\n\n"
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

    # revert model name change from test_model_print()
    local original_model_filename="granite-7b-lab-Q4_K_M.gguf"
    local temporary_model_filename='foo.gguf'
    mv "${ILAB_CACHE_DIR}/models/${temporary_model_filename}" "${ILAB_CACHE_DIR}/models/${original_model_filename}" || true
    rm -f "${ILAB_CONFIG_FILE}.bak" serve.log "${ILAB_CACHE_DIR}/models/${temporary_model_filename}" chat.log

    # reset config file and re-init defaults
    init_config
    set -e
}

########################################
# Initializes the environment necessary for this test script
# Creates a temporary directory structure for the test data,
# and sets the necessary environment variables.
#
# Arguments:
#   None
# Globals:
#   HOME
#   ILAB_DATA_DIR
#   ILAB_CONFIG_FILE
#   ILAB_CONFIGDIR_LOCATION
#   CACHE_DIR
#   DATA_DIR
#   CONFIG_DIR
# Outputs:
#   Writes logs to stdout
########################################
function init_test_script() {
    local configdir_script
    local datadir_script
    local prev_home="${HOME}"

    printf 'initializing test script...\n'
    HOME="${TEST_DIR}"  # to ensure xdg-base-dirs uses the test directory for the duration of this script
    printf 'changing home directory to "%s", was: "%s"\n' "${HOME}" "${prev_home}"

    # DO NOT REMOVE THE VARIABLE ASSIGNMENTS BELOW
    # They might seem redundant given that `mkdir -p`` is called later, but they are necessary to
    # ensure the directories are created by ilab CLI
    # get the directories from the xdg-base-dirs library
    # Get configuration, data, and cache directories using Python
    CONFIG_DIR=$(python -c "import xdg_base_dirs; print(xdg_base_dirs.xdg_config_home())")
    DATA_DIR=$(python -c "import xdg_base_dirs; print(xdg_base_dirs.xdg_data_home())")
    CACHE_DIR=$(python -c "import xdg_base_dirs; print(xdg_base_dirs.xdg_cache_home())")

    # Define package-specific directories
    configdir_script=$(printf 'import xdg_base_dirs, os; print(os.path.join(xdg_base_dirs.xdg_config_home(), "%s"))' "${PACKAGE_NAME}")
    datadir_script=$(printf 'import xdg_base_dirs, os; print(os.path.join(xdg_base_dirs.xdg_data_home(), "%s"))' "${PACKAGE_NAME}")
    cachedir_script=$(printf 'import xdg_base_dirs, os; print(os.path.join(xdg_base_dirs.xdg_cache_home(), "%s"))' "${PACKAGE_NAME}")

    ILAB_CONFIGDIR_LOCATION=$(python -c "${configdir_script}")
    ILAB_CONFIG_FILE="${ILAB_CONFIGDIR_LOCATION}/config.yaml"
    ILAB_DATA_DIR=$(python -c "${datadir_script}")
    ILAB_CACHE_DIR=$(python -c "${cachedir_script}")

    # ensure these exist at the time of running the test
    for d in "${CONFIG_DIR}" "${DATA_DIR}" "${CACHE_DIR}"; do
        printf 'creating directory: "%s"\n' "${d}"
        mkdir -p "${d}"
    done
    printf 'finished initializing test script\n'
}

########################################
# This function converts a given string to uppercase.
# Arguments:
#  A string to convert to uppercase.
# Outputs:
#  The uppercase string.
########################################
function to_upper() {
    local to_convert="$1"
    printf '%s' "${to_convert^^}"
}

wait_for_server(){
    if ! timeout 60 bash -c '
        until curl 127.0.0.1:8000|grep "{\"message\":\"Hello from InstructLab! Visit us at https://instructlab.ai\"}"; do
            echo "waiting for server to start"
            sleep 1
        done
    '; then
        echo "server did not start"
        exit 1
    fi
}


#########
# TESTS #
#########

test_oci_model_download_with_vllm_backend(){
   # Enable globstar for recursive globbing
    shopt -s globstar

    # Run the ilab model download command with REGISTRY_AUTH_FILE
    REGISTRY_AUTH_FILE=$HOME/auth.json ilab model download --repository docker://quay.io/instructlab/granite-7b-lab --release latest --model-dir models/instructlab

    patterns=(
        "models/instructlab/granite-7b-lab/config.json"
        "models/instructlab/granite-7b-lab/tokenizer.json"
        "models/instructlab/granite-7b-lab/tokenizer_config.json"
        "models/instructlab/granite-7b-lab/*.safetensors"
    )

    match_count=0
    for pattern in "${patterns[@]}"
    do
        # shellcheck disable=SC2206
        # we want to split the output into an array
        matching_files=($pattern)
        if [ ! -s "${matching_files[0]}" ]; then
            echo "No files found matching pattern: $pattern: ${matching_files[0]}"
        else
            echo "Files found matching pattern: $pattern: ${matching_files[0]}"
            match_count=$((match_count+1))
        fi
    done

    if [ $match_count -ne ${#patterns[@]} ]; then
        echo "Error: Not all files were found, only $match_count files were found"
        exit 1
    fi
}

test_llama_backend(){
    ilab model serve --backend llama-cpp &
    PID_SERVE=$!
    wait_for_server
}

# check that ilab model serve is working
test_bind_port(){
    local formatted_script
    formatted_script=$(printf '
    set timeout 60
    spawn ilab model serve
    expect {
        "http://127.0.0.1:8000/docs" { puts OK }
        default { exit 1 }
    }

    # check that ilab model serve is detecting the port is already in use
    # catch "error while attempting to bind on address ...."
    spawn ilab model serve
    expect {
        "error while attempting to bind on address " { puts OK }
        default { exit 1 }
    }

    # configure a different port
    exec sed -i.bak 's/8000/9999/g' %s

    # check that ilab model serve is working on the new port
    # catch ERROR strings in the output
    spawn ilab model serve
    expect {
        "http://127.0.0.1:9999/docs" { puts OK }
        default { exit 1 }
    }
' "${ILAB_CONFIG_FILE}")
    expect -c "${formatted_script}"
}

test_ctx_size(){
    # A context size of 55 will allow a small message
    ilab model serve --max-ctx-size 25 &> "$TEST_CTX_SIZE_LAB_SERVE_LOG_FILE" &
    PID_SERVE=$!

    # Make sure the server has time to open the port
    # if "ilab model chat" tests it before its open then it will run its own server without --max-ctx-size 1
    wait_for_server

    # SHOULD SUCCEED: ilab model chat will trim the SYS_PROMPT then take the second message
    "${chat_shot[@]}" "Hello"

    # SHOULD FAIL: ilab model chat will trim the SYS_PROMPT AND the second message, then raise an error
    # The errors from failures will be written into the serve log and chat log files
    "${chat_shot[@]}" "hello, I am a ci message that should not finish because I am too long for the context window, tell me about your day please?
    How many tokens could you take today. Could you tell me about the time you could only take twenty five tokens" &> "$TEST_CTX_SIZE_LAB_CHAT_LOG_FILE" &
    PID_CHAT=$!

    # we catch these failure BEOFRE they hit the server. as of llama_cpp_python 0.3.z, these types of failures are fatal and will crash the server
    # look for the context size error in the chat logs
    if ! timeout 20 bash -c "
        until grep -q 'Message too large for context size.' $TEST_CTX_SIZE_LAB_CHAT_LOG_FILE; do
        echo 'waiting for chat error'
        sleep 1
    done
"; then
        echo "context size error not found in chat logs"
        cat $TEST_CTX_SIZE_LAB_CHAT_LOG_FILE
        exit 1
    fi
}

test_loading_session_history(){
    ilab model serve --backend llama-cpp --max-ctx-size 128 &
    PID_SERVE=$!

    # chat with the server
    expect -c '
        set timeout 120
        spawn ilab model chat
        expect ">>>"
        send "this is a test! what are you? do not exceed ten words in your reply.\r"
        send "/s test_session_history\r"
        send "exit\r"
        expect eof
    '

    # verify the session history file was created
    if ! timeout 5 bash -c "
        until test -s test_session_history; do
        echo 'waiting for session history file to be created'
        sleep 1
    done
"; then
        echo "session history file not found"
        exit 1
    fi

    expect -c '
        spawn ilab model chat -s test_session_history
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

test_data_generate(){
    mkdir -p test_taxonomy/compositional_skills
    cat - <<EOF >  test_taxonomy/compositional_skills/qna.yaml
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

    sed -i.bak -e 's/sdg_scale_factor.*/sdg_scale_factor: 1/g' "${ILAB_CONFIG_FILE}"

    # This should be finished in a minute or so but time it out incase it goes wrong
    if ! timeout 20m ilab data generate --pipeline simple --model "${ILAB_CACHE_DIR}/models/granite-7b-lab-Q4_K_M.gguf"  --taxonomy-path test_taxonomy/compositional_skills/qna.yaml; then
        echo "Error: ilab data generate command took more than 20 minutes and was cancelled"
        exit 1
    fi

    # Test if generated was created and contains files
    local generated_dir
    generated_dir="${ILAB_DATA_DIR}/datasets"
    ls -l "${generated_dir}/"*
    if [ "$(jq ". | length" < "$(find "${generated_dir}/generated_*" | tail -n 1)")" -lt 1 ] ; then
        echo "Not enough generated results"
        exit 1
    fi
}

test_model_train() {
    # TODO: call `ilab model train`
    if [[ "$(uname)" == Linux ]]; then
        # real `ilab model train` on Linux produces models/ggml-model-f16.gguf
        # fake gml-model-f16.gguf the same as granite-7b-lab-Q4_K_M.gguf
        checkpoints_dir="${ILAB_DATA_DIR}/checkpoints/ggml-model-f16.gguf"
        test -f "${checkpoints_dir}" || \
            ln -s "${ILAB_CACHE_DIR}/models/granite-7b-lab-Q4_K_M.gguf" "${checkpoints_dir}"
    fi
}

test_model_test() {
    if [[ "$(uname)" == Linux ]]; then
        echo Using the latest of:
        ls -ltr "${ILAB_DATA_DIR}"/datasets/test_*
        timeout 20m ilab model test > model_test.out
        cat model_test.out
        grep -q '### what is 1+1' model_test.out
        grep -q 'granite-7b-lab-Q4_K_M.gguf: The sum of 1 + 1 is indeed 2' model_test.out
    fi
}

test_temp_server(){
    expect -c '
        set timeout 120
        spawn ilab model chat
        expect {
            "Trying to connect to model server at" { exit 0 }
            timeout { exit 1 }
        }
    '
}

test_temp_server_sigint(){
    expect -c '
        set timeout 120
        spawn ilab model chat
        expect "Trying to connect to model server at"
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
        set timeout 120
        spawn ilab model chat
        expect "Trying to connect to model server at"
        send "hello!\r"
        sleep 1
        expect {
            "openai._base_client" { exit 1 }
        }
        '
}

test_server_welcome_message(){
    # test that the server welcome message is displayed
    ilab model serve --log-file serve.log &
    PID_SERVE=$!

    wait_for_server

    if ! timeout 10 bash -c '
        until test -s serve.log; do
            echo "waiting for server log file to be created"
            sleep 1
        done
        if ! grep instructlab.model.backends.llama_cpp serve.log; then
            echo "server log file does not contain serving message"
            exit 1
        else
            cat serve.log
        fi
    '; then
        echo "server log file was not created"
        exit 1
    fi
}

test_log_format(){
    # remove the log 'levelname' from the log format
    sed -i.bak -e 's/log_format:.*/log_format: "%(name)s:%(lineno)d: %(message)s"/g' "${ILAB_CONFIG_FILE}"
    ilab model serve &> serve.log &
    PID_SERVE=$!

    wait_for_server

    if ! timeout 10 bash -c '
        until test -s serve.log; do
            echo "waiting for server log file to be created"
            sleep 1
        done
        if grep INFO serve.log; then
            echo "INFO log level found in server log - wrong log format"
            exit 1
        fi
    '; then
        echo "server log file was not created"
        exit 1
    fi
}

test_model_print(){
    local src_model="${ILAB_CACHE_DIR}/models/granite-7b-lab-Q4_K_M.gguf"
    local target_model="${ILAB_CACHE_DIR}/models/foo.gguf"
    cp "${src_model}" "${target_model}"
    ilab model serve --model-path "${target_model}" &
    PID_SERVE=$!

    wait_for_server

    # validate that we print the model from the server since it is different from the config
    local expected_model_name
    local expect_script
    expected_model_name=$(to_upper "${target_model}")
    expect_script=$(printf '
        spawn ilab model chat
        expect {
            timeout { exit 124 }
            -re "%s" { exit 0 }
        }
    ' "$(basename "${expected_model_name}")" # use basename otherwise the regex will not match since the path contains slashes
    )
    if ! expect -d -c "${expect_script}"; then
        echo "Error: expect test failed"
        exit 1
    fi

    # validate that we don't fail on invalid model
    if ! expect -d -c '
        set timeout 30
        spawn ilab model chat -m bar
        expect {
            timeout { exit 124 }
           -re "Requested model bar is not served by the server. Proceeding to chat with served model" { exit 0 }
        }
    '; then
        echo "Error: expect test failed"
        exit 1
    fi

    # # If we don't specify a model, validate that we print the model reported
    # # by the server since it is different from the config.
    expect_script=$(printf '
        spawn ilab model chat
        expect {
            timeout { exit 124 }
            -re "%s" { exit 0 }
        }
    ' "$(basename "${expected_model_name}")"
    )
    if ! expect -d -c"${expect_script}"; then
        echo "Error: expect test failed"
        exit 1
    fi
}

test_server_template_value(){
    export template_value="$1"
    export expected_result="$2"
    # shellcheck disable=SC2016
    expect -c '
        set template_value $env(template_value)
        set expected_result $env(expected_result)
        set timeout 10
        spawn ilab model serve --chat-template $template_value
        set actual_result 0
        expect {
            "Replacing chat template" {
                set actual_result 1
                expect {
                    "test provided custom template" {
                       set actual_result 3
                     }
                     "Starting server process" {
                     }
                }
            }
            "Starting server process" {
                set actual_result 2
            }
        }
        send \x03
        if { $actual_result != $expected_result } {
            puts stderr "Error: ilab model serve with --chat-template command failed"
            exit 1
        }
    '
}

test_server_chat_template() {
    test_server_template_value "auto" 1
    cleanup
    test_server_template_value tokenizer 2
    cleanup
    test_server_template_value "$SCRIPTDIR/test-data/mock-template.txt" 3
}

test_ilab_chat_server_logs(){
    sed -i.bak '/max_ctx_size: 4096/s/4096/10/' "${ILAB_CONFIG_FILE}"

    chat_shot+=("--serving-log-file" "chat.log")
    "${chat_shot[@]}" "Hello"

    if ! timeout 3 bash -c '
        until test -s chat.log; do
            echo "waiting for chat log file to be created"
            sleep 1
        done
        if ! grep "Message too large for context size" chat.log; then
            echo "chat log file does not contain serving message"
            exit 1
        else
            cat chat.log
        fi
    '; then
        echo "chat log file was not created"
        exit 1
    fi
}

#########
# SETUP #
#########

# shellcheck disable=SC2155
export SCRIPTDIR=$(dirname "$0")
# build a prompt string that includes the time, source file, line number, and function name
export PS4='+$(date +"%Y-%m-%d %T") ${BASH_VERSION}:${BASH_SOURCE}:${LINENO}: ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'

if [ -n "$TEST_DIR" ]; then
    echo "TEST_DIR is set to $TEST_DIR by the caller, will not remove it"
    TEST_DIR_SET_BY_CALLER=1
fi

# Support overriding the test directory for local testing otherwise creates a temporary directory
TEST_DIR=${TEST_DIR:-$(mktemp -d)}

export TEST_DIR
export PACKAGE_NAME='instructlab'  # name we use of the top-level package directories for CLI data

# get the directories from the platformdirs library
export CONFIG_DIR
export DATA_DIR
export CACHE_DIR

# we define the path here to reference elsewhere, but the existence of this file
# will be managed by the ilab CLI
export ILAB_CONFIGDIR_LOCATION
export ILAB_CONFIG_FILE
export ILAB_DATA_DIR

export TEST_CTX_SIZE_LAB_SERVE_LOG_FILE=test_ctx_size_lab_serve.log
export TEST_CTX_SIZE_LAB_CHAT_LOG_FILE=test_ctx_size_lab_chat.log

for cmd in ilab expect timeout; do
    if ! type -p $cmd; then
        echo "Error: ${cmd} is not installed"
        exit 1
    fi
done

PID_SERVE=
PID_CHAT=

chat_shot=("ilab" "model" "chat" "-qq")

init_test_script
trap 'cleanup "${?}"; test -z "${TEST_DIR_SET_BY_CALLER}" && rm -rf "${TEST_DIR}"' EXIT QUIT INT TERM

rm -f "${ILAB_CONFIG_FILE}"

# print version
ilab --version

# print system information
ilab system info

# initialize the configuration file
init_config

# download the latest version of the ilab
ilab model download

########
# MAIN #
########
# call cleanup in-between each test so they can run without conflicting with the server/chat process
test_ilab_chat_server_logs
cleanup
test_oci_model_download_with_vllm_backend
cleanup
test_bind_port
cleanup
test_ctx_size
cleanup
test_loading_session_history
cleanup
test_data_generate
cleanup
test_model_train
cleanup
test_model_test
cleanup
test_temp_server
cleanup
test_temp_server_sigint
cleanup
test_no_chat_logs
cleanup
test_server_chat_template
cleanup
test_server_welcome_message
cleanup
test_model_print
cleanup
test_log_format
cleanup
test_llama_backend
cleanup

exit 0
