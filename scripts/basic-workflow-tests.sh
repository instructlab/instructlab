#!/usr/bin/env bash
set -euf

# This is a basic workflow test of the ilab CLI.
#
# We expect it to be run anywhere `ilab` would run, including the instructlab
# container images.
#
# It represents the tasks a typical user would run through to get familiar with ilab.
#
# It is written in shell script because this basic workflow *is* a shell
# workflow, run through step by step at a shell prompt by a user.

MINIMAL=0
NUM_INSTRUCTIONS=5
GENERATE_ARGS=()
TRAIN_ARGS=()
CI=0
GRANITE=0
FULLTRAIN=0

export GREP_COLORS='mt=1;33'
BOLD='\033[1m'
NC='\033[0m' # No Color

SCRIPTDIR=$(dirname "$0")

step() {
    echo -e "$BOLD$* - $(date)$NC"
}

task() {
    echo -e "$BOLD------------------------------------------------------$NC"
    step "$@"
}

set_defaults() {
    if [ "$MINIMAL" -eq 0 ]; then
        return
    fi

    NUM_INSTRUCTIONS=1
    GENERATE_ARGS+=("--num-cpus" "$(nproc)")
    TRAIN_ARGS+=("--num-epochs" "1")
}

test_smoke() {
    task Smoke test InstructLab
    ilab | grep --color 'Usage: ilab'
}

test_init() {
    task Initializing ilab
    [ -f config.yaml ] || ilab config init --non-interactive

    step Checking config.yaml
    grep merlinite config.yaml
}

test_download() {
    task Download the model

    if [ "$GRANITE" -eq 1 ]; then
        step Downloading the granite model
        ilab model download --repository instructlab/granite-7b-lab-GGUF --filename granite-7b-lab-Q4_K_M.gguf
    else
        step Downloading the default model
        ilab model download
    fi
}

test_serve() {
    # Accepts an argument of the model, or default here
    if [ "$GRANITE" -eq 1 ]; then
        model="${1:-./models/granite-7b-lab-Q4_K_M.gguf}"
    else
        model="${1:-}"
    fi
    SERVE_ARGS=()
    if [ -n "$model" ]; then
        SERVE_ARGS+=("--model-path ${model}")
    fi

    task Serve the model
    ilab model serve "${SERVE_ARGS[@]}" &

    ret=1
    for i in $(seq 1 10); do
        sleep 5
    	step "$i"/10: Waiting for model to start
        if curl -sS http://localhost:8000/docs > /dev/null; then
            ret=0
            break
        fi
    done

    return $ret
}

test_chat() {
    task Chat with the model
    CHAT_ARGS=()
    if [ "$GRANITE" -eq 1 ]; then
        CHAT_ARGS+=("-m models/granite-7b-lab-Q4_K_M.gguf")
    fi
    printf 'Say "Hello"\n' | ilab model chat "${CHAT_ARGS[@]}" | grep --color 'Hello'
}

test_taxonomy() {
    task Update the taxonomy

    TESTNUM=$1
    if [ "$TESTNUM" -ne 1 ] && [ "$TESTNUM" -ne 2 ] && [ "$TESTNUM" -ne 3 ]; then
        echo "Invalid test number: $TESTNUM"
        exit 1
    fi

    test -d taxonomy || git clone https://github.com/instructlab/taxonomy || true

    step Update taxonomy with sample qna additions
    if [ "$TESTNUM" -eq 1 ]; then
        mkdir -p taxonomy/compositional_skills/extraction/inference/qualitative/e2e-siblings
        cp "$SCRIPTDIR"/test-data/e2e-qna-freeform-skill.yaml taxonomy/compositional_skills/extraction/inference/qualitative/e2e-siblings/qna.yaml
    elif [ "$TESTNUM" -eq 2 ]; then
        rm -rf taxonomy/compositional_skills/extraction/inference/qualitative/e2e-siblings
        mkdir -p taxonomy/compositional_skills/extraction/answerability/e2e-yes_or_no
        cp "$SCRIPTDIR"/test-data/e2e-qna-grounded-skill.yaml taxonomy/compositional_skills/extraction/answerability/e2e-yes_or_no/qna.yaml
    elif [ "$TESTNUM" -eq 3 ]; then
        rm -rf taxonomy/compositional_skills/extraction/answerability/e2e-yes_or_no
        mkdir -p taxonomy/knowledge/sports/overview/e2e-softball
        cp "$SCRIPTDIR"/test-data/e2e-qna-knowledge.yaml taxonomy/knowledge/sports/overview/e2e-softball/qna.yaml
    fi

    step Verification
    ilab taxonomy diff
}

test_generate() {
    task Generate synthetic data
    if [ "$GRANITE" -eq 1 ]; then
        GENERATE_ARGS+=("--model ./models/granite-7b-lab-Q4_K_M.gguf")
    fi
    ilab data generate --num-instructions ${NUM_INSTRUCTIONS} "${GENERATE_ARGS[@]}"
}

test_train() {
    task Train the model

    # TODO Only cuda for now
    TRAIN_ARGS=("--legacy" "--device=cuda")
    if [ "$FULLTRAIN" -eq 0 ]; then
        TRAIN_ARGS+=("--4-bit-quant")
    fi
    if [ "$GRANITE" -eq 1 ]; then
        TRAIN_ARGS+=("--gguf-model-path models/granite-7b-lab-Q4_K_M.gguf")
    fi

    ilab model train "${TRAIN_ARGS[@]}"
}

test_convert() {
    task Converting the trained model and serving it
    ilab model convert
}

test_exec() {
    # The list of actual tests to run through in workflow order
    test_smoke
    test_init
    test_download

    # See below for cleanup, this runs an ilab model serve in the background
    test_serve
    PID=$!

    test_chat

    test_taxonomy 1
    test_generate
    test_taxonomy 2
    test_generate
    test_taxonomy 3
    test_generate

    # Kill the serve process
    task Stopping the ilab model serve
    step Kill ilab model serve $PID
    kill $PID

    test_train

    if [ "$CI" -eq 1 ]; then
        # This is what we expect to work in our CI runner so far
        return
    fi

    # The rest is TODO when we can make it work on our CI runner

    # When you run this --
    #   `ilab model convert` is only implemented for macOS with M-series chips for now
    #test_convert

    # TODO: chat with the new model
    test_serve /tmp/somemodelthatispretrained.gguf
    PID=$!

    # TODO: Ask a qestion about softball
    test_chat

    # Kill the serve process
    task Stopping the ilab model serve
    step Kill ilab model serve $PID
    kill $PID
}

usage() {
    echo "Usage: $0 [-m] [-h]"
    echo "  -m  Run minimal configuration (run quicker when you have no GPU)"
    echo "  -c  Run in CI mode (explicitly skip steps we know will fail in linux CI)"
    echo "  -f  Run the fullsize training instead of --4-bit-quant"
    echo "  -g  Use the granite model"
    echo "  -h  Show this help text"

}

# Process command line arguments
task "Configuring ..."
while getopts "cmfgh" opt; do
    case $opt in
        c)
            CI=1
            step "Running in CI mode."
            ;;
        m)
            MINIMAL=1
            step "Running minimal configuration."
            ;;
        f)
            FULLTRAIN=1
            step "Running fullsize training."
            ;;
        g)
            GRANITE=1
            step "Running with granite model."
            ;;
        h)
            usage
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            exit 1
            ;;
    esac
done

set_defaults
test_exec
