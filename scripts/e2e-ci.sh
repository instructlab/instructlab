#!/usr/bin/env bash
set -xeuf

# This is the full workflow test of the ilab CLI.
#
# We expect it to be run anywhere `ilab` would run, including the instructlab
# container images.
#
# It represents the workflow a typical user would run through with ilab.

# generic globals
BOLD='\033[1m'
NC='\033[0m' # No Color
PRESERVE=0

# path and token globals
SCRIPTDIR=$(dirname "$0")
E2E_TEST_DIR=""
CONFIG_HOME=""
DATA_HOME=""
CACHE_HOME=""
CONFIG_HOME=""
TRAINED_MODEL_PATH=""
HF_TOKEN=${HF_TOKEN:-}

# model globals
GRANITE_7B_MODEL="instructlab/granite-7b-lab"
MIXTRAL_8X7B_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
PROMETHEUS_8X7B_MODEL="prometheus-eval/prometheus-8x7b-v2.0"
MERLINITE_GGUF_MODEL="merlinite-7b-lab-Q4_K_M.gguf"

# t-shirt size globals
SMALL=0
MEDIUM=0
LARGE=0

check_flags() {
    if [ "${SMALL}" -ne 1 ] && [ "${MEDIUM}" -ne 1 ] && [ "${LARGE}" -ne 1 ]; then
         echo "ERROR: Must specify a size flag when invoking this script."
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
    task InstructLab smoke test
    ilab | grep --color 'Usage: ilab'
    task InstructLab smoke test Complete
}

test_system_info() {
    task Output system info
    ilab system info
    task System info Complete
}

test_init() {
    task Initializing InstructLab
    if [ "$SMALL" -eq 1 ]; then
        # We do not use a train profile here because the 'full' train pipeline does not use a GPU
        ilab config init --non-interactive

        step Setting small-size SDG-specific config
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" generate.pipeline simple
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" generate.teacher.model_path "${CACHE_HOME}/instructlab/models/${MERLINITE_GGUF_MODEL}"

        step Setting small-size Training-specific config
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.ckpt_output_dir "${DATA_HOME}/instructlab/checkpoints"
        #TODO(nweinber): Uncomment below line when 'pipeline' is added to train config
        #python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.pipeline simple
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.model_path "${CACHE_HOME}/instructlab/models/${GRANITE_7B_MODEL}"
        #TODO(nweinber): Uncomment below line when 'device' is added to train config
        #python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.device cuda
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.num_epochs 1
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.effective_batch_size 4

        # We do not set any Eval-specific config here as the 'small' E2E workflow does not run evaluation

    elif [ "$MEDIUM" -eq 1 ]; then
        # We do not use a train profile here because the 'full' train pipeline does not use a GPU
        ilab config init --non-interactive

        step Setting medium-size SDG-specific config
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" generate.pipeline simple
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" generate.teacher.model_path "${CACHE_HOME}/instructlab/models/${MERLINITE_GGUF_MODEL}"

        step Setting medium-size Training-specific config
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.ckpt_output_dir "${DATA_HOME}/instructlab/checkpoints"
        #TODO(nweinber): Uncomment below line when 'pipeline' is added to train config
        #python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.pipeline full
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.model_path "${CACHE_HOME}/instructlab/models/${GRANITE_7B_MODEL}"
        #TODO(nweinber): Uncomment below line when 'device' is added to train config
        #python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.device cpu
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.num_epochs 1
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.effective_batch_size 4

        step Setting medium-size Eval-specific config
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" evaluate.gpus 1
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" evaluate.mt_bench.judge_model "${CACHE_HOME}/instructlab/models/${GRANITE_7B_MODEL}"

    elif [ "$LARGE" -eq 1 ]; then
        step Initalizing ilab with l40s-x4 train profile
        ilab config init --non-interactive --train-profile="${SCRIPTDIR}/test-data/train-profile-l40s-x4.yaml"

        step Setting large-size Serve-specific config
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" serve.vllm.gpus 4

        step Setting large-size SDG-specific config
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" generate.pipeline full
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" generate.teacher.model_path "${CACHE_HOME}/instructlab/models/${MIXTRAL_8X7B_MODEL}"
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" generate.teacher.vllm.gpus 4
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" generate.teacher.vllm.llm_family mixtral

        step Setting large-size Training-specific config
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.phased_mt_bench_judge "${CACHE_HOME}/instructlab/models/${PROMETHEUS_8X7B_MODEL}"
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.model_path "${CACHE_HOME}/instructlab/models/${GRANITE_7B_MODEL}"
        #TODO(nweinber): Uncomment below line when 'device' is added to train config
        #python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" train.device cuda

        step Setting large-size Eval-specific config
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" evaluate.gpus 4
        python "${SCRIPTDIR}"/e2e_config_edit.py  "${CONFIG_HOME}/instructlab/config.yaml" evaluate.mt_bench.judge_model "${CACHE_HOME}/instructlab/models/${PROMETHEUS_8X7B_MODEL}"
    fi
    task InstructLab initialization Complete
}

test_config_show() {
    task Output active config
    ilab config show
    task Config show Complete
}

test_download() {
    task Download models
    if [ "$SMALL" -eq 1 ] || [ "$MEDIUM" -eq 1 ]; then
        step Downloading the merlinite-7b-lab GGUF model as the teacher model for SDG
        ilab model download --repository instructlab/merlinite-7b-lab-GGUF --filename ${MERLINITE_GGUF_MODEL}
        step Downloading granite-7b-lab model to train and as the judge model for evaluation
        ilab model download --repository ${GRANITE_7B_MODEL}
    elif [ "$LARGE" -eq 1 ]; then
        step Downloading the mixtral-8x7b instruct model as the teacher model for SDG
        ilab model download --repository ${MIXTRAL_8X7B_MODEL} --hf-token "${HF_TOKEN}"
        step Downloading the prometheus-8x7b model as the judge model for evaluation
        ilab model download --repository ${PROMETHEUS_8X7B_MODEL} --hf-token "${HF_TOKEN}"
        step Downloading granite-7b-lab model to train
        ilab model download --repository ${GRANITE_7B_MODEL}
    fi
    task Downloading models Complete
}

test_list() {
    task List the Downloaded Models
    if [ "$MEDIUM" -eq 1 ]; then
        ilab model list | grep ${GRANITE_7B_MODEL}
        ilab model list | grep ${MERLINITE_GGUF_MODEL}
    elif [ "$LARGE" -eq 1 ]; then
        ilab model list | grep ${GRANITE_7B_MODEL}
        ilab model list | grep ${PROMETHEUS_8X7B_MODEL}
        ilab model list | grep ${MIXTRAL_8X7B_MODEL}
    fi
    task Model Listing Complete
}

test_taxonomy() {
    task Update the taxonomy

    step Add compositional skill to the taxomony
    mkdir -p "$DATA_HOME"/instructlab/taxonomy/compositional_skills/extraction/inference/qualitative/e2e-siblings
    cp "$SCRIPTDIR"/test-data/e2e-qna-freeform-skill.yaml "$DATA_HOME"/instructlab/taxonomy/compositional_skills/extraction/inference/qualitative/e2e-siblings/qna.yaml

    if [ "$LARGE" -eq 1 ]; then
        step Add knowledge to the taxomony
        mkdir -p "$DATA_HOME"/instructlab/taxonomy/knowledge/phoenix/overview/e2e-phoenix
        cp "$SCRIPTDIR"/test-data/e2e-qna-knowledge-phoenix.yaml "$DATA_HOME"/instructlab/taxonomy/knowledge/phoenix/overview/e2e-phoenix/qna.yaml

        step Add grounded skill to the taxomony
        mkdir -p "$DATA_HOME"/instructlab/taxonomy/compositional_skills/extraction/answerability/e2e-yes_or_no
        cp "$SCRIPTDIR"/test-data/e2e-qna-grounded-skill.yaml "$DATA_HOME"/instructlab/taxonomy/compositional_skills/extraction/answerability/e2e-yes_or_no/qna.yaml
    fi

    step Verify the taxonomy
    ilab taxonomy diff

    task Taxonomy updates and verification Complete
}

test_generate() {
    task Generate synthetic data
    ilab data generate
    task Synthetic data generation Complete
}

test_data_list() {
    task Output data list
    ilab data list
    task Data list Complete
}

test_train() {
    task Train the model

    local knowledge_data_path
    local skill_data_path
    skill_data_path=$(find "${DATA_HOME}"/instructlab/datasets -name 'skills_train_msgs*' | head -n 1)
    knowledge_data_path=$(find "${DATA_HOME}"/instructlab/datasets -name 'knowledge_train_msgs*' | head -n 1)
    #TODO(nweinber): Replace with commented-out lines below when 'init' is fixed
    if [ "$SMALL" -eq 1 ]; then
        ilab model train --4-bit-quant --device=cuda --pipeline=simple
        #ilab model train --4-bit-quant
    elif [ "$MEDIUM" -eq 1 ]; then
        ilab model train --data-path "${skill_data_path}" --device=cpu --pipeline=full
        #ilab model train --data-path "${skill_data_path}"
    fi

    TRAINED_MODEL_PATH="${DATA_HOME}/instructlab/checkpoints/hf_format/samples_0"

    task Model training Complete
}

test_phased_train() {
    task Train the model with LAB multi-phase training

    # 'Declare and assign separately to avoid masking return values' <-- error.
    local knowledge_data_path
    local skills_data_path
    knowledge_data_path=$(find "${DATA_HOME}"/instructlab/datasets -name 'knowledge_train_msgs*' | head -n 1)
    skills_data_path=$(find "${DATA_HOME}"/instructlab/datasets -name 'skills_train_msgs*' | head -n 1)

    # general phased training args
    local train_args
    train_args+=(
        "--strategy=lab-multiphase"
        "--phased-phase1-data=${knowledge_data_path}"
        "--phased-phase2-data=${skills_data_path}"
        "--skip-user-confirm"
    )

    export INSTRUCTLAB_EVAL_MMLU_MIN_TASKS=true
    export HF_DATASETS_TRUST_REMOTE_CODE=true
    ilab model train "${train_args[@]}" 2>&1 | tee "${E2E_TEST_DIR}"/multiphase_training.log

    # final best model is written to log
    TRAINED_MODEL_PATH=$(grep "Training finished! Best final checkpoint: " "${E2E_TEST_DIR}"/multiphase_training.log | grep -o "/[^ ]*")
    task Multi-phase training Complete
}

test_serve() {
    task Serve the model

    local model_path
    # When we run training with --4-bit-quant, we can't convert the result to a gguf
    # https://github.com/instructlab/instructlab/issues/579
    # so we skip trying to test the result and just ensure that serve works in general
    if [ "$SMALL" -eq 1 ]; then
        model_path="${CACHE_HOME}/instructlab/models/${MERLINITE_GGUF_MODEL}"
    # use trained gguf for medium-size job
    elif [ "$MEDIUM" -eq 1 ]; then
        model_path="${TRAINED_MODEL_PATH}/pytorch_model-Q4_K_M.gguf"
    # use safetensors for large-size job
    elif [ "$LARGE" -eq 1 ]; then
        model_path="${TRAINED_MODEL_PATH}"
    fi

    # output serve log while `ilab model serve` is starting up
    touch serve.log
    tail -f serve.log &
    TPID=$!

    ilab model serve --model-path "${model_path}" &> serve.log &
    wait_for_server start
    kill "$TPID"
    task Model serving Complete
}

test_chat() {
    task Chat with the model
    ilab model chat -qq 'Say "Hello" and nothing else\n'
    task Chat with the model Complete
}

test_evaluate() {
    task Evaluate the model

    local model_path
    local base_model_path

    model_path=${TRAINED_MODEL_PATH}
    base_model_path="${CACHE_HOME}/instructlab/models/${GRANITE_7B_MODEL}"

    export INSTRUCTLAB_EVAL_MMLU_MIN_TASKS=true
    export HF_DATASETS_TRUST_REMOTE_CODE=true
    step Running MMLU
    ilab model evaluate --model "${model_path}" --benchmark mmlu

    if [ "$LARGE" -eq 1 ]; then
        step Running MMLU Branch
        local mmlu_branch_tasks_dir
        mmlu_branch_tasks_dir=$(find "${DATA_HOME}"/instructlab/datasets -name 'node_datasets_*' | head -n 1)
        ilab model evaluate --model "${model_path}" --benchmark mmlu_branch --tasks-dir "${mmlu_branch_tasks_dir}" --base-model "${base_model_path}"
    fi

    export INSTRUCTLAB_EVAL_FIRST_N_QUESTIONS=20
    step Running MT Bench
    ilab model evaluate --model "${model_path}" --benchmark mt_bench

    step Running MT Bench Branch
    cd "${DATA_HOME}/instructlab/taxonomy"
    git branch rc
    ilab model evaluate \
        --model "${model_path}" \
        --branch rc \
        --base-branch main \
        --base-model "${base_model_path}" \
        --benchmark mt_bench_branch \
        --taxonomy-path "${DATA_HOME}/instructlab/taxonomy"

    task Evaluation Complete
}

test_exec() {
    # smoke tests
    test_smoke
    test_system_info

    # intitalization tests
    test_init
    test_config_show

    # model download tests
    test_download
    test_list

    # sdg tests
    test_taxonomy
    test_generate
    test_data_list

    # train tests
    if [ "$LARGE" -eq 1 ]; then
      test_phased_train
    else
      test_train
    fi

    # serve + chat tests
    test_serve
    PID=$!
    test_chat

    # kill the serve process
    task Stopping the ilab model serve for trained model
    wait_for_server shutdown $PID

    # evaluate tests
    # note we do not run eval on the small runner due to a lack of GPU resources
    if [ "$SMALL" -eq 0 ]; then
        test_evaluate
    fi

    task E2E success!
    exit 0
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
    echo "Usage: $0 [-s] [-m] [-l] [-h]"
    echo "  -s  Run small t-shirt size job"
    echo "  -m  Run medium t-shirt size job"
    echo "  -l  Run large t-shirt size job"
    echo "  -p  Preserve the E2E_TEST_DIR for debugging"
    echo "  -h  Show this help text"
}

# Process command line arguments
task "Configuring ..."
while getopts "smlph" opt; do
    case $opt in
        s)
            SMALL=1
            step "Run small T-shirt size job."
            ;;
        m)
            MEDIUM=1
            step "Run medium T-shirt size job."
            ;;
        l)
            LARGE=1
            step "Run large T-shirt size job."
            ;;
        p)
            PRESERVE=1
            step "Preserve the E2E_TEST_DIR for debugging."
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
if [ "$PRESERVE" -eq 0 ]; then
    trap 'rm -rf "${E2E_TEST_DIR}"' EXIT
fi
test_exec
