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
MIXTRAL=0
NUM_INSTRUCTIONS=5
GENERATE_ARGS=("--num-cpus" "$(nproc)" --taxonomy-path='./taxonomy')
DIFF_ARGS=("--taxonomy-path" "./taxonomy")
TRAIN_ARGS=()
GRANITE=0
FULLTRAIN=0
BACKEND="llama-cpp"
HF_TOKEN=${HF_TOKEN:-}
SDG_PIPELINE="simple"
SKIP_TRAIN=${SKIP_TRAIN:-0}
EVAL=0

export GREP_COLORS='mt=1;33'
BOLD='\033[1m'
NC='\033[0m' # No Color

SCRIPTDIR=$(dirname "$0")
export E2E_TEST_DIR
# E2E_TEST_DIR="$(pwd)/__e2e_test"
export CONFIG_HOME
export DATA_HOME
export CONFIG_HOME



########################################
# This function initializes the directory
# used during end-to-end testing. This function
# also creates the directory structure which
# instructlab expects to see at the time of execution.
# Arguments:
#   None
# Globals:
#   DATA_HOME
#   CONFIG_HOME
#   STATE_HOME
#   E2E_TEST_DIR
#   HOME
# Outputs:
#   None
########################################
function init_e2e_tests() {
    E2E_TEST_DIR=$(mktemp -d)
    HOME="${E2E_TEST_DIR}"  # update the HOME directory used to resolve paths

    CONFIG_HOME=$(python -c 'import platformdirs; print(platformdirs.user_config_dir())')
    DATA_HOME=$(python -c 'import platformdirs; print(platformdirs.user_data_dir())')
    STATE_HOME=$(python -c 'import platformdirs; print(platformdirs.user_state_dir())')
    # ensure that our mock e2e dirs exist
    for dir in "${CONFIG_HOME}" "${DATA_HOME}" "${STATE_HOME}"; do
    	mkdir -p "${dir}"
    done
}



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

    # Minimal settings to run in less time
    NUM_INSTRUCTIONS=1
    TRAIN_ARGS+=("--num-epochs" "1")

    if [ "${GRANITE}" -eq 1 ] && [ "${MIXTRAL}" -eq 1 ]; then
        echo "ERROR: Can not specify -g and -M at the same time."
        exit 1
    fi

    if [ "${MIXTRAL}" -eq 1 ] && [ "${BACKEND}" = "vllm" ]; then
        echo "ERROR: Can not specify -M and -v at the same time."
        exit 1
    fi

    if [ "${MIXTRAL}" -eq 1 ] && [ -z "${HF_TOKEN}" ]; then
        echo "ERROR: Must specify HF_TOKEN env var to download mixtral."
        exit 1
    fi
}

test_smoke() {
    task Smoke test InstructLab
    ilab | grep --color 'Usage: ilab'
}

test_init() {
    task Initializing ilab
    ilab config init --non-interactive

    step Checking config.yaml
    if [ "${MIXTRAL}" -eq 1 ]; then
        sed -i -e 's|merlinite.*|mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf|' "${CONFIG_HOME}/instructlab/config.yaml"
    fi
}

test_download() {
    task Download the model

    if [ "$GRANITE" -eq 1 ]; then
        step Downloading the granite model
        ilab model download --repository instructlab/granite-7b-lab-GGUF --filename granite-7b-lab-Q4_K_M.gguf
    elif [ "$BACKEND" = "vllm" ]; then
        step Downloading the model for vLLM
        ilab download --repository instructlab/merlinite-7b-lab
    elif [ "$MIXTRAL" -eq 1 ]; then
        step Downloading the mixtral model
        ilab model download --repository TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF --filename mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf --hf-token "${HF_TOKEN}"
    else
        step Downloading the default model
        ilab model download
    fi

    if [ "$EVAL" -eq 1 ]; then
        step Downloading a model to use with evaluation
        ilab model download --repository instructlab/granite-7b-lab
    fi
}

test_serve() {
    # Accepts an argument of the model, or default here
    if [ "$GRANITE" -eq 1 ]; then
        model="${1:-${DATA_HOME}/instructlab/models/granite-7b-lab-Q4_K_M.gguf}"
    elif [ "${MIXTRAL}" -eq 1 ]; then
        model="${1:-${DATA_HOME}/instructlab/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf}"
    else
        model="${1:-}"
    fi
    SERVE_ARGS=()
    if [ -n "$model" ]; then
        SERVE_ARGS+=("--model-path" "${model}")
    fi
    if [ "$BACKEND" = "vllm" ]; then
        SERVE_ARGS+=("--model-path" "${DATA_HOME}/instructlab/models/instructlab/merlinite-7b-lab")
    fi

    task Serve the model
    ilab model serve "${SERVE_ARGS[@]}" &> serve.log &
    wait_for_server
}

test_chat() {
    task Chat with the model
    CHAT_ARGS=()
    if [ "$GRANITE" -eq 1 ]; then
        CHAT_ARGS+=("-m" "${DATA_HOME}/instructlab/models/granite-7b-lab-Q4_K_M.gguf")
    elif [ "$MIXTRAL" -eq 1 ]; then
        CHAT_ARGS+=("-m" "${DATA_HOME}/instructlab/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf" "--model-family" "mixtral")
    fi
    printf 'Say "Hello" and nothing else\n' | ilab model chat -qq "${CHAT_ARGS[@]}"
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
        mkdir -p taxonomy/knowledge/tonsils/overview/e2e-tonsils
        cp "$SCRIPTDIR"/test-data/e2e-qna-knowledge.yaml taxonomy/knowledge/tonsils/overview/e2e-tonsils/qna.yaml
    fi

    step Verification
    ilab taxonomy diff "${DIFF_ARGS[@]}"
}

test_generate() {
    task Generate synthetic data
    if [ "$GRANITE" -eq 1 ]; then
        GENERATE_ARGS+=("--model" "${DATA_HOME}/instructlab/models/granite-7b-lab-Q4_K_M.gguf")
    elif [ "$MIXTRAL" -eq 1 ]; then
        GENERATE_ARGS+=("--model" "${DATA_HOME}/instructlab/models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf")
    elif [ "$BACKEND" = "vllm" ]; then
        GENERATE_ARGS+=("--model" "${DATA_HOME}/instructlab/models/instructlab/merlinite-7b-lab")
    fi
    if [ "$SDG_PIPELINE" = "full" ]; then
        GENERATE_ARGS+=("--pipeline" "full")
    fi
    ilab data generate --num-instructions ${NUM_INSTRUCTIONS} "${GENERATE_ARGS[@]}"
}

test_train() {
    task Train the model

    # TODO Only cuda for now
    TRAIN_ARGS+=("--legacy" "--device=cuda")
    if [ "$FULLTRAIN" -eq 0 ]; then
        TRAIN_ARGS+=("--4-bit-quant")
    fi
    if [ "$GRANITE" -eq 1 ]; then
        TRAIN_ARGS+=("--gguf-model-path" "${DATA_HOME}/instructlab/models/granite-7b-lab-Q4_K_M.gguf")
    fi

    ilab model train "${TRAIN_ARGS[@]}"
}

test_convert() {
    task Converting the trained model and serving it
    ilab model convert
}

test_evaluate() {
    task Evaluate the model

    if [ "$EVAL" -eq 1 ]; then
      export INSTRUCTLAB_EVAL_MMLU_MIN_TASKS=true
      export HF_DATASETS_TRUST_REMOTE_CODE=true
      ilab model evaluate --model "${DATA_HOME}/instructlab/models/instructlab/granite-7b-lab" --benchmark mmlu
    fi
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

    if [ "$SKIP_TRAIN" -eq 1 ]; then
        # TODO - Drop this later.
        # This is only a temporary measure while we bootstrap different CI workflows.
        # There are some larger environments where it only makes sense to test the new
        # training workflow using the training library, but that is not yet integrated
        # here. Skip training for those environments for now.
        task Halting prior to running training "(SKIP_TRAIN=1)"
        return
    fi

    test_train

    if [ "$FULLTRAIN" -eq 0 ]; then
        # When we run training with --4-bit-quant, we can't convert the result to a gguf
        # https://github.com/instructlab/instructlab/issues/579
        # so we skip trying to test the result
        return
    fi

    # When you run this --
    #   `ilab model convert` is only implemented for macOS with M-series chips for now
    #test_convert

    test_serve "${DATA_HOME}/instructlab/models/ggml-model-f16.gguf"
    PID=$!

    test_chat

    # Kill the serve process
    task Stopping the ilab model serve
    step Kill ilab model serve $PID
    kill $PID

    task Evaluating the output of ilab model train
    test_evaluate
}

wait_for_server(){
    if ! timeout 120 bash -c '
        until curl -sS http://localhost:8000/docs &> /dev/null; do
            echo "waiting for server to start"
            sleep 1
        done
    '; then
        echo "server did not start"
        cat serve.log || true
        exit 1
    fi
    echo "server started"
}

usage() {
    echo "Usage: $0 [-m] [-h]"
    echo "  -m  Run minimal configuration (run quicker when you have no GPU)"
    echo "  -e  Run model evaluation"
    echo "  -f  Run the fullsize training instead of --4-bit-quant"
    echo "  -F  Use the 'full' SDG pipeline instead of the default 'simple' pipeline"
    echo "  -g  Use the granite model"
    echo "  -v  Use the vLLM backend for serving"
    echo "  -M  Use the mixtral model (4-bit quantized)"
    echo "  -h  Show this help text"
}

# Process command line arguments
task "Configuring ..."
while getopts "cemMfFghv" opt; do
    case $opt in
        c)
            # old option, don't fail if it's specified
            ;;
        e)
            EVAL=1
            step "Run model evaluation."
            ;;
        m)
            MINIMAL=1
            step "Running minimal configuration."
            ;;
        M)
            MIXTRAL=1
            step "Using mixtral model (4-bit quantized)."
            ;;
        f)
            FULLTRAIN=1
            step "Running fullsize training."
            ;;
        F)
            SDG_PIPELINE="full"
            step "Using full SDG pipeline."
            ;;
        g)
            GRANITE=1
            step "Running with granite model."
            ;;
        h)
            usage
            exit 0
            ;;
        v)
            BACKEND=vllm
            step "Running with vLLM backend."
            ;;
        \?)
            echo "Invalid option: -$opt" >&2
            usage
            exit 1
            ;;
    esac
done

init_e2e_tests
trap 'rm -rf "${E2E_TEST_DIR}"' EXIT
set_defaults
test_exec
