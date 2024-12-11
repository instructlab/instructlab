#!/usr/bin/env bash
set -xeuf

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
GENERATE_ARGS=("--endpoint-url" "http://localhost:8000/v1" "--num-cpus" "$(nproc)" --taxonomy-path='./taxonomy')
DIFF_ARGS=("--taxonomy-path" "./taxonomy")
TRAIN_ARGS=()
PHASED_TRAINING=0
BACKEND="llama-cpp"
FOUR_BIT_QUANT=0
SIMPLE_TRAIN=0
FULL_TRAIN=0
ACCELERATED_TRAIN=0
HF_TOKEN=${HF_TOKEN:-}
SDG_PIPELINE="full"
SKIP_TRAIN=${SKIP_TRAIN:-0}
EVAL=0
MIXTRAL_GGUF_MODEL="mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
GRANITE_GGUF_MODEL="granite-7b-lab-Q4_K_M.gguf"
MERLINITE_GGUF_MODEL="merlinite-7b-lab-Q4_K_M.gguf"
GRANITE_SAFETENSOR_REPO="instructlab/granite-7b-lab"

export GREP_COLORS='mt=1;33'
BOLD='\033[1m'
NC='\033[0m' # No Color

SCRIPTDIR=$(dirname "$0")
export E2E_TEST_DIR
# E2E_TEST_DIR="$(pwd)/__e2e_test"
export CONFIG_HOME
export DATA_HOME
export CACHE_HOME
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
#   CACHE_HOME
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
    CACHE_HOME=$(python -c 'import platformdirs; print(platformdirs.user_cache_dir())')
    STATE_HOME=$(python -c 'import platformdirs; print(platformdirs.user_state_dir())')
    # ensure that our mock e2e dirs exist
    for dir in "${CONFIG_HOME}" "${DATA_HOME}" "${STATE_HOME}" "${CACHE_HOME}"; do
    	mkdir -p "${dir}"
    done

    E2E_LOG_DIR="${HOME}/log"
    mkdir -p "${E2E_LOG_DIR}"
}



step() {
    echo -e "$BOLD$* - $(date)$NC"
}

task() {
    echo -e "$BOLD------------------------------------------------------$NC"
    step "$@"
}

set_defaults() {
    if [ "${EVAL}" -eq 1 ] && [ "${BACKEND}" != "vllm" ]; then
         echo "ERROR: Must specify -e and -v at the same time."
         exit 1
    fi

    if [ "${MIXTRAL}" -eq 1 ] && [ -z "${HF_TOKEN}" ]; then
        echo "ERROR: Must specify HF_TOKEN env var to download mixtral."
        exit 1
    fi

    if [ "${PHASED_TRAINING}" -eq 1 ] && [ "${ACCELERATED_TRAIN}" -eq 0 ]; then
        echo "ERROR: You have -P set. It requires -a."
        exit 1
    fi

    if [ "$MINIMAL" -eq 1 ]; then
        # Minimal settings to run in less time
        NUM_INSTRUCTIONS=1
    fi
}

test_smoke() {
    task InstructLab smoke test
    ilab | grep --color 'Usage: ilab'
    task InstructLab smoke test Complete
}

test_init() {
    task Initializing ilab
    
    ilab config init --non-interactive

    step Replace model in config.yaml
    if [ "${BACKEND}" == "vllm" ]; then
        sed -i -e "s|merlinite.*|${GRANITE_SAFETENSOR_REPO}|" "${CONFIG_HOME}/instructlab/config.yaml"
    else
        sed -i -e "s|merlinite.*|${GRANITE_GGUF_MODEL}|" "${CONFIG_HOME}/instructlab/config.yaml"
    fi
}

test_download() {
    task Download models

    if [ "${BACKEND}" == "vllm" ]; then
        step Downloading the model granite .safetensors model
        ilab model download --repository ${GRANITE_SAFETENSOR_REPO}
    else
        step Downloading the granite .gguf model
        ilab model download --repository instructlab/granite-7b-lab-GGUF --filename ${GRANITE_GGUF_MODEL}
    fi

    if [ "$MIXTRAL" -eq 1 ]; then
        step Downloading the mixtral model as the teacher model for SDG
        ilab model download --repository TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF --filename ${MIXTRAL_GGUF_MODEL} --hf-token "${HF_TOKEN}"
    else
        step Downloading the default merlinite .gguf model as the teacher model for SDG
        ilab model download --repository instructlab/merlinite-7b-lab-GGUF --filename ${MERLINITE_GGUF_MODEL}
    fi

    if [ "$EVAL" -eq 1 ]; then
        step Downloading merlinite .safetensors model to use in evaluation as the judge model
        ilab model download --repository instructlab/merlinite-7b-lab
    fi

    task Downloading models Complete
}

test_list() {
    task List the Downloaded Models

    if [ "$GRANITE" -eq 1 ]; then
        step Listing the granite GGUF model only
        ilab model list | grep granite-7b-lab-Q4_K_M.gguf
    elif [ "$MIXTRAL" -eq 1 ]; then
        step Listing the mixtral model only
        ilab model list | grep mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf
    else
        step Listing the merlinite GGUF model only
        ilab model list | grep merlinite-7b-lab-Q4_K_M.gguf4
    fi
    
    # regardless, download merl safetensors to test that capability
    ilab model download --repository instructlab/merlinite-7b-lab
    ilab model list | grep instructlab/merlinite-7b-lab
}

test_serve() {
    local model_type
    local serve_args
    local model_path

    model_type="$1"
    serve_args=()
    if [ "${model_type}" == "base" ]; then
        if [ "${BACKEND}" == "vllm" ]; then
            model_path="${CACHE_HOME}/instructlab/models/${GRANITE_SAFETENSOR_REPO}"
        else    # if not using vLLM, use the GGUF for llama-cpp
            model_path="${CACHE_HOME}/instructlab/models/${GRANITE_GGUF_MODEL}"
        fi
    elif [ "$model_type" == "teacher" ]; then
        if [ "${MIXTRAL}" -eq 1 ]; then
            model_path="${CACHE_HOME}/instructlab/models/${MIXTRAL_GGUF_MODEL}"
            serve_args+=("--model-family" "mixtral")
        else
            model_path="${CACHE_HOME}/instructlab/models/${MERLINITE_GGUF_MODEL}"
        fi
    elif [ "$model_type" == "trained" ]; then
        model_path="${2:-}"
    else
        echo "Model type of 'teacher', 'base', or 'trained' not passed as an arg of test_serve()"
        exit 1
    fi

    if [ -n "$model_path" ]; then
        serve_args+=("--model-path" "${model_path}")
    fi

    task Serve the model

    # output serve log while `ilab model serve` is starting up
    touch serve.log
    tail -f serve.log &
    TPID=$!

    ilab model serve "${serve_args[@]}" &> serve.log &
    wait_for_server start
    kill "$TPID"
}

test_chat() {
    task Chat with the model
    CHAT_ARGS=("--endpoint-url" "http://localhost:8000/v1")
    ilab model chat -qq "${CHAT_ARGS[@]}" 'Say "Hello" and nothing else\n'
}

test_taxonomy() {
    task Update the taxonomy
    local testnum

    testnum=$1
    if [ "$testnum" -ne 1 ] && [ "$testnum" -ne 2 ] && [ "$testnum" -ne 3 ]; then
        echo "Invalid test number: $testnum"
        exit 1
    fi

    test -d taxonomy || git clone https://github.com/instructlab/taxonomy || true

    step Update taxonomy with sample qna additions
    if [ "$testnum" -eq 1 ]; then
        mkdir -p taxonomy/compositional_skills/extraction/inference/qualitative/e2e-siblings
        cp "$SCRIPTDIR"/test-data/e2e-qna-freeform-skill.yaml taxonomy/compositional_skills/extraction/inference/qualitative/e2e-siblings/qna.yaml
    elif [ "$testnum" -eq 2 ]; then
        rm -rf taxonomy/compositional_skills/extraction/inference/qualitative/e2e-siblings
        mkdir -p taxonomy/compositional_skills/extraction/answerability/e2e-yes_or_no
        cp "$SCRIPTDIR"/test-data/e2e-qna-grounded-skill.yaml taxonomy/compositional_skills/extraction/answerability/e2e-yes_or_no/qna.yaml
    elif [ "$testnum" -eq 3 ]; then
        rm -rf taxonomy/compositional_skills/extraction/answerability/e2e-yes_or_no
        mkdir -p taxonomy/knowledge/tonsils/overview/e2e-tonsils
        cp "$SCRIPTDIR"/test-data/knowledge/e2e-qna-knowledge.yaml taxonomy/knowledge/tonsils/overview/e2e-tonsils/qna.yaml
    fi

    step Verification
    ilab taxonomy diff "${DIFF_ARGS[@]}"
}

test_generate() {
    task Generate synthetic data

    if [ "$MIXTRAL" -eq 1 ]; then
        GENERATE_ARGS+=("--model" "${CACHE_HOME}/instructlab/models/${MIXTRAL_GGUF_MODEL}")
    else
        GENERATE_ARGS+=("--model" "${CACHE_HOME}/instructlab/models/${MERLINITE_GGUF_MODEL}")
    fi

    # default arg for '--pipeline' is "full"
    if [ "$SDG_PIPELINE" = "simple" ]; then
        GENERATE_ARGS+=("--pipeline" "simple")
    fi

    # Disable batching with llama-cpp. See https://github.com/instructlab/instructlab/issues/1892
    if [ "$BACKEND" = "llama-cpp" ]; then
        GENERATE_ARGS+=("--batch-size" "0")
    fi

    # NUM_INSTRUCTIONS is '1' if MINIMAL is set, '5' otherwise
    ilab data generate --num-instructions ${NUM_INSTRUCTIONS} "${GENERATE_ARGS[@]}"

    task Synthetic data generation Complete
}

test_train() {
    task Train the model

    local data
    data=$(find "${DATA_HOME}"/instructlab/datasets -name 'skills_train_msgs_*' | head -n 1)

    # simple, full, and accelerated, are different workflows
    # To mimic a real user e2e scenario, only one of these should be run on a given system
    # The `small` worker can manage `simple`, The medium worker can handle `full` and the large worker can handle `accelerated`
    if [ "$ACCELERATED_TRAIN" -eq 1 ]; then
        # TODO Only cuda for now
        # the train profile specified in test_init overrides the majority of TRAIN_ARGS, including things like num_epochs. While it looks like much of those settings are being lost, they just have different values here.
        TRAIN_ARGS=("--pipeline=accelerated" "--device=cuda" "--model-path=${GRANITE_SAFETENSOR_REPO}" "--data-path=${data}" "--lora-quantize-dtype=nf4" "--4-bit-quant" "--effective-batch-size=4" "--is-padding-free=False")
        if [ "${BACKEND}" != "vllm" ]; then
            TRAIN_ARGS+=("--gguf-model-path" "${CACHE_HOME}/instructlab/models/${GRANITE_GGUF_MODEL}")
        fi
    fi
    if [ "$SIMPLE_TRAIN" -eq 1 ]; then
        if [ "$FOUR_BIT_QUANT" -eq 1 ]; then
            TRAIN_ARGS+=("--4-bit-quant")
        fi
        # TODO Only cuda for now
        TRAIN_ARGS+=("--pipeline=simple" "--device=cuda" "--num-epochs=1")
        if [ "${BACKEND}" != "vllm" ]; then
            TRAIN_ARGS+=("--gguf-model-path" "${CACHE_HOME}/instructlab/models/${GRANITE_GGUF_MODEL}")
        fi
    fi
    if [ "$FULL_TRAIN" -eq 1 ]; then
        # test training on a CPU not the GPU
        TRAIN_ARGS=("--num-epochs=1" "--pipeline=full" "--model-path=${GRANITE_SAFETENSOR_REPO}" "--data-path=${data}" "--effective-batch-size=4" --device=cpu)
    fi

    ilab model train "${TRAIN_ARGS[@]}"
}

test_phased_train() {
    task Train the model with LAB multi-phase strategy.

    # linter wants data_path and other variables declared then assigned separately.
    # 'Declare and assign separately to avoid masking return values' <-- error.
    local data_path
    data_path=$(find "${DATA_HOME}"/instructlab/datasets -name 'messages_*' | head -n 1)

    local model_path
    model_path="${CACHE_HOME}/instructlab/models/${GRANITE_SAFETENSOR_REPO}"

    # general training args - not the same as the global TRAIN_ARGS
    local train_args
    train_args=(
        "--model-path=${model_path}"
        "--lora-quantize-dtype=nf4"
        "--4-bit-quant"
        "--is-padding-free=False"
        "--effective-batch-size=4"
    )

    # general phased training args
    train_args+=(
        "--phased-training"
        "--skip-user-confirm"
        "--phased-mt-bench-judge-dir=${model_path}" # uses base model
    )

    # phase 1 args 
    train_args+=(
        "--phased-phase1-num-epochs=1"
        "--phased-phase1-data=${data_path}"
        "--phased-phase1-samples-per-save=1"
    )

    # phase 2 args 
    train_args+=(
        "--phased-phase2-num-epochs=1"
        "--phased-phase2-data=${data_path}"
        "--phased-phase2-samples-per-save=1"
    )

    ilab model train "${train_args[@]}"
    # final best model is written to stdout.

}

test_convert() {
    task Converting the trained model and serving it
    ilab model convert
}

test_evaluate() {
    task Evaluate the model
    
    local model_path="${CACHE_HOME}/instructlab/models/${GRANITE_SAFETENSOR_REPO}"
    # Temporarily using merlinite as the base model to confirm the workflow executes correctly 
    local base_model_path="${CACHE_HOME}/instructlab/models/instructlab/merlinite-7b-lab"

    export INSTRUCTLAB_EVAL_MMLU_MIN_TASKS=true
    export HF_DATASETS_TRUST_REMOTE_CODE=true
    task Running MMLU
    ilab model evaluate --model "${model_path}" --benchmark mmlu
    task Running MMLU_BRANCH
    ilab model evaluate --model "${model_path}" --benchmark mmlu_branch --tasks-dir tests/testdata/mmlu_branch/ --base-model "${base_model_path}"

    export INSTRUCTLAB_EVAL_FIRST_N_QUESTIONS=5
    task Running MT_Bench
    ilab model evaluate --model "${model_path}" --judge-model "${model_path}" --enable-serving-output --benchmark mt_bench
    task Running MT_Bench_Branch
    cd "${DATA_HOME}/instructlab/taxonomy"
    git branch rc
    cd "${DATA_HOME}/instructlab"
    GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    ilab model evaluate \
    --model "${model_path}" \
    --judge-model "${model_path}" \
    --branch rc --base-branch main \
    --base-model "${base_model_path}" \
    --gpus "${GPUS}" \
    --enable-serving-output \
    --benchmark mt_bench_branch \
    --taxonomy-path "${DATA_HOME}/instructlab/taxonomy"

}

test_system_info() {
    task Output system info
    ilab system info
    task System info output complete
}

test_config_show() {
    task Output active config
    ilab config show
    task Config show Complete
}

test_data_list() {
    task Output data list
    ilab data list
    task Data list Complete
}

test_exec() {
    # The list of actual tests to run through in workflow order
    test_smoke
    test_system_info
    test_init
    test_config_show
    test_download

    # See below for cleanup, this runs an ilab model serve in the background
    test_serve base
    PID=$!

    test_chat

    task Stopping the ilab model serve for the base model
    wait_for_server shutdown $PID

    test_serve teacher
    PID=$!
    step served the teacher model

    test_taxonomy 1
    test_generate
    test_taxonomy 2
    test_generate
    test_taxonomy 3
    test_generate

    test_data_list

    # Kill the serve process
    task Stopping the ilab model serve for the teacher model
    wait_for_server shutdown $PID

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

    # When we run training with --4-bit-quant, we can't convert the result to a gguf
    # https://github.com/instructlab/instructlab/issues/579
    # so we skip trying to test the result
    if [ "$FULL_TRAIN" -eq 1 ]; then
        # When you run this --
        #   `ilab model convert` is only implemented for macOS with M-series chips for now
        #test_convert

        # when using full train, choose any GGUF from any of the checkpoints dirs
        model_dir=$(find "${DATA_HOME}"/instructlab/checkpoints/hf_format -name 'samples_*' | head -n 1)

        test_serve trained "${model_dir}/pytorch_model-Q4_K_M.gguf"
        PID=$!

        ilab model chat -qq --model "${model_dir}/pytorch_model-Q4_K_M.gguf" --endpoint-url http://localhost:8000/v1 'Say "Hello" and nothing else\n'

        # Kill the serve process
        task Stopping the ilab model serve for trained model
        wait_for_server shutdown $PID
    fi

    if [ "$EVAL" -eq 1 ]; then
        test_evaluate
    fi

    if [ "${PHASED_TRAINING}" -eq 1 ]; then
        test_phased_train
    fi
}

wait_for_server() {
    ACTION="$1"
    if [ "${ACTION}" == "start" ]; then
      step Wait for ilab model serve to start
      CMD="curl -sS http://localhost:8000/docs &> /dev/null"
      ACTION="started"
    elif [ "${ACTION}" == "shutdown" ]; then
      PID="${2:-}"
      step Kill ilab model serve "$PID"
      kill "$PID"
      CMD="! ps -p ${PID} &> /dev/null"
    else
        echo "Action 'start' or 'shutdown' not passed as an arg of wait_for_server()"
        exit 1
    fi

    if ! timeout 240 bash -c "
        until ${CMD}; do
            echo 'waiting for server'
            sleep 5
        done
    "; then
        echo "server was not ${ACTION}"
        exit 1
    fi

    echo "server ${ACTION}"
}

# NOTE: If you add additional or modify existing options, please document them in 'docs/ci.md'
usage() {
    echo "Usage: $0 [-m] [-h]"
    echo "  -e  Run model evaluation"
    echo "  -q  Use 4-bit-quant when training"
    echo "  -a  Use the 'full' training library rather than legacy training"
    echo "  -s  Run the simple training using the SFTTainer rather than the custom training loop"
    echo "  -f  Run the fullsize training instead of --4-bit-quant"
    echo "  -S  Use the 'simple' SDG pipeline instead of the default 'full' pipeline"
    echo "  -h  Show this help text"
    echo "  -m  Run minimal configuration with lower number of instructions and training epochs (run quicker when you have no GPU)"
    echo "  -M  Use the mixtral model (4-bit quantized) instead of merlinite model (4-bit quantized)."
    echo "  -P  Run multi-phase training"
    echo "  -v  Use the vLLM backend for serving"
}

# Process command line arguments
task "Configuring ..."
while getopts "eShqasfmMPv" opt; do
    case $opt in
        e)
            EVAL=1
            step "Run model evaluation."
            ;;
        S)
            SDG_PIPELINE="simple"
            step "Using simple SDG pipeline."
            ;;
        h)
            usage
            exit 0
            ;;
        m)
            MINIMAL=1
            step "Running minimal configuration."
            ;;
        M)
            MIXTRAL=1
            step "Using mixtral model (4-bit quantized) instead of merlinite model (4-bit quantized)."
            ;;
        P)
            PHASED_TRAINING=1
            step "Running multi-phase training."
            ;;
        v)
            BACKEND=vllm
            step "Running with vLLM backend."
            ;;
        q)
            FOUR_BIT_QUANT=1
            step "Running training using 4-bit-quantization."
            ;;
        s)
            SIMPLE_TRAIN=1
            step "Running the simple training pipeline"
            ;;
        f)
            FULL_TRAIN=1
            step "Running the full training pipeline"
            ;;
        a)
            ACCELERATED_TRAIN=1
            step "Running using the training library"
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
