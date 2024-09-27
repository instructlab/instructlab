#!/usr/bin/env bash
set -xeuf

# This is the full workflow test of the ilab CLI.
#
# We expect it to be run anywhere `ilab` would run, including the instructlab
# container images.
#
# It represents the workflow a typical user would run through with ilab.

HF_TOKEN=${HF_TOKEN:-}
GRANITE_7B_MODEL="instructlab/granite-7b-lab"
MIXTRAL_8X7B_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
PROMETHEUS_8X7B_MODEL="prometheus-eval/prometheus-8x7b-v2.0"
TRAINED_MODEL_PATH=""
LARGE=0

BOLD='\033[1m'
NC='\033[0m' # No Color

SCRIPTDIR=$(dirname "$0")
E2E_TEST_DIR=""
CONFIG_HOME=""
DATA_HOME=""
CACHE_HOME=""
CONFIG_HOME=""

check_flags() {
    if [ "${LARGE}" -ne 1 ]; then
         echo "ERROR: Must specify a flag when invoking this script."
         usage
         exit 1
    fi

}

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
#   E2E_TEST_DIR
#   HOME
# Outputs:
#   None
########################################

init_e2e_tests() {
    E2E_TEST_DIR=$(mktemp -d)
    HOME="${E2E_TEST_DIR}"  # update the HOME directory used to resolve paths

    CONFIG_HOME=$(python -c 'import platformdirs; print(platformdirs.user_config_dir())')
    DATA_HOME=$(python -c 'import platformdirs; print(platformdirs.user_data_dir())')
    CACHE_HOME=$(python -c 'import platformdirs; print(platformdirs.user_cache_dir())')
    # ensure that our mock e2e dirs exist
    for dir in "${CONFIG_HOME}" "${DATA_HOME}" "${CACHE_HOME}"; do
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

test_smoke() {
    task InstructLab Smoke Test
    ilab | grep --color 'Usage: ilab'
    task InstructLab Smoke Test Complete
}

test_init() {
    task Initializing ilab

    step Setting train-profile for GPU accelerated training

    if [ "$LARGE" -eq 1 ]; then
      ilab config init --non-interactive --train-profile="${SCRIPTDIR}/test-data/train-profile-l40sx4.yaml"

      # setting large size eval specific config
      python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" evaluate.gpus 4
      python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" evaluate.mt_bench.judge_model "${CACHE_HOME}/instructlab/models/${PROMETHEUS_8X7B_MODEL}"

      # setting large size SDG specific config
      python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" generate.pipeline full
      python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" generate.teacher.model_path "${CACHE_HOME}/instructlab/models/${MIXTRAL_8X7B_MODEL}"
      python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" generate.teacher.vllm.gpus 4
      python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" generate.teacher.vllm.llm_family mixtral

      # setting large size training specific config
      python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.ckpt_output_dir "${CACHE_HOME}/instructlab/checkpoints"
      python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.phased_mt_bench_judge "${CACHE_HOME}/instructlab/models/${PROMETHEUS_8X7B_MODEL}"
      python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.model_path "${CACHE_HOME}/instructlab/models/${GRANITE_7B_MODEL}"
    fi
    task ilab Initializing Complete
}

test_download() {
    task Download models
    if [ "$LARGE" -eq 1 ]; then
        step Downloading the mixtral 8x7b instruct model as the teacher model for SDG
        ilab model download --repository ${MIXTRAL_8X7B_MODEL} --hf-token "${HF_TOKEN}"
        step Downloading the prometheus-8x7b model as judge model for evaluation
        ilab model download --repository ${PROMETHEUS_8X7B_MODEL} --hf-token "${HF_TOKEN}"
        step Downloading granite-7b-lab model to train
        ilab model download --repository instructlab/granite-7b-lab
    fi
    task Downloading Models Complete
}

test_list() {
    task List the Downloaded Models

    if [ "$LARGE" -eq 1 ]; then
        # step Listing the models for the large size use case
        ilab model list | grep ${GRANITE_7B_MODEL}
        ilab model list | grep ${PROMETHEUS_8X7B_MODEL}
        ilab model list | grep ${MIXTRAL_8X7B_MODEL}
    fi
    task Model Listing Complete
}

test_serve() {
    task Serve the model
    local model_path
    local serve_args

    serve_args=()
    model_path="${1:-}"
    if [ -n "$model_path" ]; then
        serve_args+=("--model-path" "${model_path}")
    fi


    # output serve log while `ilab model serve` is starting up
    touch serve.log
    tail -f serve.log &
    TPID=$!

    ilab model serve "${serve_args[@]}" &> serve.log &
    wait_for_server start
    kill "$TPID"
    task Model Serving Complete
}

test_chat() {
    task Chat with the model
    CHAT_ARGS=("--endpoint-url" "http://localhost:8000/v1")
    ilab model chat -m "${TRAINED_MODEL_PATH}" -qq "${CHAT_ARGS[@]}" 'Say "Hello" and nothing else\n'
    task Chat with the model complete
}

test_taxonomy() {
    task Update the taxonomy

    # add compositional skill to the taxomony
    mkdir -p "$DATA_HOME"/instructlab/taxonomy/compositional_skills/extraction/inference/qualitative/e2e-siblings
    cp "$SCRIPTDIR"/test-data/e2e-qna-freeform-skill.yaml "$DATA_HOME"/instructlab/taxonomy/compositional_skills/extraction/inference/qualitative/e2e-siblings/qna.yaml

    # add grounded skill to the taxomony
    mkdir -p "$DATA_HOME"/instructlab/taxonomy/compositional_skills/extraction/answerability/e2e-yes_or_no
    cp "$SCRIPTDIR"/test-data/e2e-qna-grounded-skill.yaml "$DATA_HOME"/instructlab/taxonomy/compositional_skills/extraction/answerability/e2e-yes_or_no/qna.yaml

    # add knowledge to the taxomony
    mkdir -p "$DATA_HOME"/instructlab/taxonomy/knowledge/phoenix/overview/e2e-phoenix
    cp "$SCRIPTDIR"/test-data/e2e-qna-knowledge-phoenix.yaml "$DATA_HOME"/instructlab/taxonomy/knowledge/phoenix/overview/e2e-phoenix/qna.yaml

    step Taxonomy Verification
    ilab taxonomy diff
    task Taxonomy Updated and Verification complete
}

test_generate() {
    task Generate synthetic data
    ilab data generate
    task Synthetic Data Generation Complete
}

test_train() {
    task Train the model
    local knowledge_data_path
    knowledge_data_path=$(find "${DATA_HOME}"/instructlab/datasets -name 'knowledge_train_msgs*' | head -n 1)
    ilab model train --data-path "${knowledge_data_path}"
    TRAINED_MODEL_PATH=$(find "${DATA_HOME}"/instructlab/checkpoints/hf_format/ -name 'samples_*' | head -n 1)
    task Model Training Complete
}

test_phased_train() {
    task Train the model with LAB multi-phase training

    # 'Declare and assign separately to avoid masking return values' <-- error.
    local knowledge_data_path
    local skills_data_path
    knowledge_data_path=$(find "${DATA_HOME}"/instructlab/datasets -name 'knowledge_train_msgs*' | head -n 1)
    skills_data_path=$(find "${DATA_HOME}"/instructlab/datasets -name 'skills_train_msgs*' | head -n 1)

    # general training args - not the same as the global TRAIN_ARGS
    local train_args

    # general phased training args
    train_args+=(
        "--strategy=lab-multiphase"
        "--phased-phase1-data=${knowledge_data_path}"
        "--phased-phase2-data=${skills_data_path}"
        "--skip-user-confirm"
    )

    if [ "$LARGE" -eq 1 ]; then
      export INSTRUCTLAB_EVAL_MMLU_MIN_TASKS=true
      export HF_DATASETS_TRUST_REMOTE_CODE=true
    fi
    ilab model train "${train_args[@]}" 2>&1 | tee "${E2E_TEST_DIR}"/multiphase_training.log
    # final best model is written to log.
    TRAINED_MODEL_PATH=$(grep "Training finished! Best final checkpoint: " "${E2E_TEST_DIR}"/multiphase_training.log | grep -o "/[^ ]*")
    task Multi-phase Training Complete
}

test_evaluate() {
    task Evaluate the model
    
    local model_path
    local base_model_path
    local judge_model_path
    local mmlu_branch_tasks_dir

    model_path=${TRAINED_MODEL_PATH}
    base_model_path="${CACHE_HOME}/instructlab/models/${GRANITE_7B_MODEL}"
    judge_model_path="${CACHE_HOME}/instructlab/models/${PROMETHEUS_8X7B_MODEL}"
    mmlu_branch_tasks_dir=$(find "${DATA_HOME}"/instructlab/datasets -name 'node_datasets_*' | head -n 1)

    if [ "$LARGE" -eq 1 ]; then
      export INSTRUCTLAB_EVAL_MMLU_MIN_TASKS=true
      export HF_DATASETS_TRUST_REMOTE_CODE=true
    fi
    task Running MMLU
    ilab model evaluate --model "${model_path}" --benchmark mmlu
    task Running MMLU_BRANCH
    ilab model evaluate --model "${model_path}" --benchmark mmlu_branch --tasks-dir "${mmlu_branch_tasks_dir}" --base-model "${base_model_path}"

    export INSTRUCTLAB_EVAL_FIRST_N_QUESTIONS=20
    task Running MT_Bench
    GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    ilab model evaluate --model "${model_path}" --judge-model "${judge_model_path}" --benchmark mt_bench --gpus "${GPUS}"
    task Running MT_Bench_Branch
    cd "${DATA_HOME}/instructlab/taxonomy"
    git branch rc
    ilab model evaluate \
    --model "${model_path}" \
    --judge-model "${judge_model_path}" \
    --branch rc --base-branch main \
    --base-model "${base_model_path}" \
    --gpus "${GPUS}" \
    --benchmark mt_bench_branch \
    --taxonomy-path "${DATA_HOME}/instructlab/taxonomy"

    task Evaluation Complete
}

test_config_show() {
    task Output active config
    ilab config show
    task Config show Complete
}

test_exec() {
    # The list of actual tests to run through in workflow order
    test_smoke

    test_init
    test_config_show

    test_download
    test_list

    test_taxonomy
    test_generate

    if [ "$LARGE" -eq 1 ]; then
      test_phased_train
    else
      test_train
    fi

    test_serve "${TRAINED_MODEL_PATH}"
    PID=$!
    test_chat

    # Kill the serve process
    task Stopping the ilab model serve for trained model
    wait_for_server shutdown $PID

    test_evaluate
    task Success
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
    echo "Usage: $0 [-l] [-h]"
    echo "  -l  Run large t-shirt size job"
    echo "  -h  Show this help text"
}

# Process command line arguments
task "Configuring ..."
while getopts "lh" opt; do
    case $opt in
        l)
            LARGE=1
            step "Run large T-shirt size job."
            ;;
        h)
            usage
            exit 0
            ;;
        \?)
            echo "Invalid option: -$opt" >&2
            usage
            exit 1
            ;;
    esac
done

check_flags
init_e2e_tests
trap 'rm -rf "${E2E_TEST_DIR}"' EXIT
test_exec
