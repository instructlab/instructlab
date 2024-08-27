#!/usr/bin/env -S bash -e
# SPDX-License-Identifier: Apache-2.0

COMMON_OPTS="hn:"
MAX_ITERS=12
INSTRUCTLAB_CLOUD_CONFIG=${INSTRUCTLAB_CLOUD_CONFIG:-$HOME/.instructlab/cloud-config}

if test -s "$INSTRUCTLAB_CLOUD_CONFIG" ; then
    # shellcheck source=/dev/null
    . "$INSTRUCTLAB_CLOUD_CONFIG"
else
    echo "Missing required config: $INSTRUCTLAB_CLOUD_CONFIG"
    exit 1
fi

ec2() {
    local cmdname=$1; shift
    if [ "$(type -t "ec2__$cmdname")" = "function" ] >/dev/null 2>&1; then
        INSTANCE_NAME="$EC2_INSTANCE_NAME"
        "ec2__$cmdname" "$@"
    elif [ "$(type -t "${cmdname//-/_}")" = "function" ] >/dev/null 2>&1; then
        INSTANCE_NAME="$EC2_INSTANCE_NAME"
        handle_help_and_instance_name_opts "$@"
        "${cmdname//-/_}" "ec2"
    else
        echo "Invalid command: $cmdname"
        show_usage
        exit 1
    fi
}

ec2__sync() {
    local temp_commit=false
    while getopts "${COMMON_OPTS}c" opt; do
        handle_common_opts "$opt" "$OPTARG"
        case "$opt" in
            c)  temp_commit=true
            ;;
        esac
    done
    ec2_calculate_instance_public_dns
    sync "ec2" "$EC2_KEY_LOCATION" "$temp_commit"
}

ec2__launch() {
    while getopts "${COMMON_OPTS}t:" opt; do
        handle_common_opts "$opt" "$OPTARG"
        case "$opt" in
            t)  EC2_INSTANCE_TYPE=$OPTARG
            ;;
        esac
    done

    local instance_id
    instance_id="$(aws ec2 run-instances \
        --image-id "$EC2_AMI_ID" \
        --region "$EC2_REGION" \
        --instance-type "$EC2_INSTANCE_TYPE" \
        --security-group-ids "$EC2_SECURITY_GROUP_ID" \
        --subnet-id "$EC2_SUBNET_ID" \
        --key-name "$EC2_KEY_NAME" \
        --block-device-mappings '{"DeviceName": "/dev/sda1","Ebs": {"VolumeSize": 300}}' \
        --associate-public-ip-address \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
        "ResourceType=volume,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
        --query 'Instances[0].InstanceId' \
        --output text)" &> /dev/null

    if [ -z "$instance_id" ]; then
        echo "Failed to launch instance"
        exit 1
    else
        local i=0
        echo "Waiting for instance to be in the 'running' state..."
        while [[ $i -lt $MAX_ITERS ]]; do
            local state
            state="$(aws ec2 describe-instances \
                    --instance-ids "$instance_id" \
                    --region "$EC2_REGION" \
                    --query 'Reservations[0].Instances[0].State.Name' \
                    --output text)" &> /dev/null 
            if [ "$state" != "running" ]; then
                sleep 5
                ((i++))
            else
                echo "Instance named ${INSTANCE_NAME}(${instance_id}) is now running"
                break
            fi
        done

        if [ $i -eq $MAX_ITERS ]; then
            echo "Gave up waiting for instance named ${INSTANCE_NAME}(${instance_id}) to start running"
        fi
    fi

}

ec2__stop() {
    handle_help_and_instance_name_opts "$@"
    ec2_calculate_instance_id
    aws ec2 stop-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$EC2_REGION"
}

ec2__start() {
    handle_help_and_instance_name_opts "$@"
    ec2_calculate_instance_id
    aws ec2 start-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$EC2_REGION"
}

ec2__terminate() {
    handle_help_and_instance_name_opts "$@"
    ec2_calculate_instance_id
    aws ec2 terminate-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$EC2_REGION"
}

wait_ssh_listen() {
    calculate_cloud_public_dns "$1"
    user_name=$(instance_user_name "$1")
    ssh_key=$(instance_key "$1")
    while true; do
        echo "Waiting for ssh..."
        sleep 5
        if ssh -i "$ssh_key" -o ConnectTimeout=5 -o StrictHostKeyChecking=no -q "$user_name"@"$PUBLIC_DNS" "true"; then
            break
        fi
    done
}

ec2__ssh() {
    handle_help_and_instance_name_opts "$@"
    shift $((OPTIND-1))

    local user_name
    user_name=$(instance_user_name "ec2")
    ec2_calculate_instance_public_dns
    ssh -o "StrictHostKeyChecking no" -i "$EC2_KEY_LOCATION" "$user_name"@"$PUBLIC_DNS" "$@"
}

ec2__scp() {
    handle_help_and_instance_name_opts "$@"
    shift $((OPTIND-1))

    local user_name
    user_name=$(instance_user_name "ec2")
    ec2_calculate_instance_public_dns
    scp -o "StrictHostKeyChecking no" -i "$EC2_KEY_LOCATION" "$@" "$user_name"@"$PUBLIC_DNS":
}

ec2_calculate_instance_id() {
    if [ -z "$INSTANCE_ID" ]; then
        INSTANCE_ID="$(aws ec2 describe-instances \
            --filters "Name=tag:Name,Values=$INSTANCE_NAME" \
            --region "$EC2_REGION" \
            --query "Reservations[*].Instances[*].InstanceId" \
            --output text)" &> /dev/null
    fi
    if [ -z "$INSTANCE_ID" ]; then
        echo "Instance named '${INSTANCE_NAME}' not found"
        exit 1
    fi
}

ec2_calculate_instance_public_dns() {
    ec2_calculate_instance_id
    PUBLIC_DNS="$(aws ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$EC2_REGION" \
        --query "Reservations[*].Instances[*].PublicDnsName" \
        --output text)" &> /dev/null
    PUBLIC_DNS=$(echo "$PUBLIC_DNS" | xargs)
}

ibm() {
    local cmdname=$1; shift
    if [ "$(type -t "ibm__$cmdname")" = "function" ] >/dev/null 2>&1; then
        INSTANCE_NAME="$IBM_INSTANCE_NAME"
        "ibm__$cmdname" "$@"
    elif [ "$(type -t "${cmdname//-/_}")" = "function" ] >/dev/null 2>&1; then
        INSTANCE_NAME="$IBM_INSTANCE_NAME"
        handle_help_and_instance_name_opts "$@"
        "${cmdname//-/_}" "ibm"
    else
        echo "Invalid command: $cmdname"
        show_usage
        exit 1
    fi
}

ibm__stop() {
    handle_help_and_instance_name_opts "$@"
    ibmcloud is instance-stop -f "$INSTANCE_NAME"
}

ibm__start() {
    handle_help_and_instance_name_opts "$@"
    ibmcloud is instance-start -f "$INSTANCE_NAME"
}

ibm__terminate() {
    handle_help_and_instance_name_opts "$@"
    ibmcloud is instance-delete -f "$INSTANCE_NAME"
}

ibm__sync() {
    local temp_commit=false
    while getopts "${COMMON_OPTS}c" opt; do
        handle_common_opts "$opt" "$OPTARG"
        case "$opt" in
            c)  temp_commit=true
            ;;
        esac
    done
    ibm_calculate_instance_public_dns
    sync "ibm" "$IBM_KEY_LOCATION" "$temp_commit"
}

ibm__launch() {
    while getopts "${COMMON_OPTS}t:" opt; do
        handle_common_opts "$opt" "$OPTARG"
        case "$opt" in
            t)  IBM_INSTANCE_PROFILE_NAME="$OPTARG"
            ;;
        esac
    done

    ibmcloud is instance-create "$INSTANCE_NAME" "$IBM_VPC_ID" "$IBM_ZONE" "$IBM_INSTANCE_PROFILE_NAME" "$IBM_SUBNET_ID" --image "$IBM_IMAGE_ID" --boot-volume '{"name": "boot-vol-attachment-name", "volume": {"name": "boot-vol", "capacity": 200, "profile": {"name": "general-purpose"}}}' --keys "$IBM_KEY_NAME" --pnac-vni-name "$INSTANCE_NAME"

    ibmcloud is virtual-network-interface-floating-ip-add "$INSTANCE_NAME" "$INSTANCE_NAME" "$IBM_FLOATING_IP_NAME"

    local i
    i=0
    echo "Waiting for instance to be in the 'running' state..."
    while [[ $i -lt $MAX_ITERS ]]; do
        local state
        state="$(ibmcloud is instance "${INSTANCE_NAME}" --output JSON | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")"
        if [ "$state" != "running" ]; then
            sleep 5
            ((i++))
        else
            echo "Instance named '${INSTANCE_NAME}' is now running"
            break
        fi
    done

    if [ $i -eq $MAX_ITERS ]; then
        echo "Gave up waiting for instance named '${INSTANCE_NAME}' to start running"
    fi
}

ibm__ssh() {
    handle_help_and_instance_name_opts "$@"
    shift $((OPTIND-1))

    local user_name
    user_name="$(instance_user_name "ibm")"
    ibm_calculate_instance_public_dns
    ssh -o "StrictHostKeyChecking no" -i "$IBM_KEY_LOCATION" "$user_name"@"$PUBLIC_DNS" "$@"
}

ibm_calculate_instance_public_dns() {
    PUBLIC_DNS=$(ibmcloud is instance "$INSTANCE_NAME" --output json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['network_interfaces'][0]['floating_ips'][0]['address'])")
}

calculate_cloud_public_dns() {
    local cloud_type=$1
    if [ "$cloud_type" = 'ec2' ]; then
	ec2_calculate_instance_public_dns
    elif [ "$cloud_type" = 'ibm' ]; then
        ibm_calculate_instance_public_dns
    fi
}

instance_key() {
    local cloud_type=$1
    if [ "$cloud_type" = 'ec2' ]; then
        echo "$EC2_KEY_LOCATION"
    elif [ "$cloud_type" = 'ibm' ]; then
        echo "$IBM_KEY_LOCATION"
    fi
}

instance_user_name() {
    local cloud_type=$1
    if [ "$cloud_type" = 'ec2' ]; then
        echo "ec2-user"
    elif [ "$cloud_type" = 'ibm' ]; then
        echo "root"
    fi
}

instance_user_home() {
    local cloud_type=$1
    if [ "$cloud_type" = 'ec2' ]; then
        echo "/home/ec2-user"
    elif [ "$cloud_type" = 'ibm' ]; then
        echo "/root"
    fi
}

setup_rh_devenv() {
    local cloud_type=$1
    "${BASH_SOURCE[0]}" "$cloud_type" ssh -n "$INSTANCE_NAME" sudo dnf install git gcc make python3.11 python3.11-devel -y
    "${BASH_SOURCE[0]}" "$cloud_type" ssh -n "$INSTANCE_NAME" "sudo dnf install g++ -y || sudo dnf install gcc-c++"
    "${BASH_SOURCE[0]}" "$cloud_type" ssh -n "$INSTANCE_NAME" "if [ ! -d instructlab.git ]; then git clone --bare https://github.com/instructlab/instructlab.git && git clone instructlab.git && pushd instructlab && git remote add syncrepo ../instructlab.git && git remote add upstream https://github.com/instructlab/instructlab.git; fi"
    "${BASH_SOURCE[0]}" "$cloud_type" ssh -n "$INSTANCE_NAME" "pushd instructlab && python3.11 -m venv --upgrade-deps venv"
}

pip_install_with_nvidia() {
    local cloud_type=$1
    "${BASH_SOURCE[0]}" "$cloud_type" ssh -n "$INSTANCE_NAME" \
        "pushd instructlab && sed 's/\[.*\]//' requirements.txt > constraints.txt"
    "${BASH_SOURCE[0]}" "$cloud_type" ssh -n "$INSTANCE_NAME" \
        "pushd instructlab && source venv/bin/activate; \
         export PATH=\$PATH:/usr/local/cuda/bin; \
         pip cache remove llama_cpp_python \
         && CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install --force-reinstall --no-binary \
            llama_cpp_python -c constraints.txt llama_cpp_python \
         && pip install wheel packaging torch -c constraints.txt \
         && pip install .[cuda] -r requirements-vllm-cuda.txt \
         && ilab"
}

pip_install_with_amd() {
    local cloud_type=$1
    "${BASH_SOURCE[0]}" "$cloud_type" ssh -n "$INSTANCE_NAME" "pushd instructlab && source venv/bin/activate && pip cache remove llama_cpp_python && \
    pip install -e .[rocm] --extra-index-url https://download.pytorch.org/whl/rocm6.0 \
   -C cmake.args='-DLLAMA_HIPBLAS=on' \
   -C cmake.args='-DAMDGPU_TARGETS=all' \
   -C cmake.args='-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang' \
   -C cmake.args='-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++' \
   -C cmake.args='-DCMAKE_PREFIX_PATH=/opt/rocm'"
}

update_rh_nvidia_drivers() {
    install_rh_nvidia_drivers "$1" "true"
}

install_rh_nvidia_drivers() {
    local cloud_type=$1
    local should_update=$2
    if [[ "$should_update" == "true" ]]; then
        "${BASH_SOURCE[0]}" "$cloud_type" ssh -n "$INSTANCE_NAME" sudo dnf update -y
        "${BASH_SOURCE[0]}" "$cloud_type" ssh -n "$INSTANCE_NAME" sudo reboot
        echo "Rebooting instance after update.."
        wait_ssh_listen "$cloud_type"
        "${BASH_SOURCE[0]}" "$cloud_type" ssh -n "$INSTANCE_NAME" sudo dnf remove --oldinstallonly -y
    fi
    "${BASH_SOURCE[0]}" "$cloud_type" scp -n "$INSTANCE_NAME" nvidia-setup.sh
    "${BASH_SOURCE[0]}" "$cloud_type" ssh -n "$INSTANCE_NAME" "sudo ./nvidia-setup.sh"
    echo "You may want to reboot even though the install is live (${BASH_SOURCE[0]} ${cloud_type} ssh sudo reboot)"
}

sync() {
    local cloud_type=$1
    local key_location=$2
    local temp_commit=$3
    local user_name
    user_name="$(instance_user_name "$cloud_type")"
    local user_home
    user_home="$(instance_user_home "$cloud_type")"

    local branch
    branch="$(git symbolic-ref HEAD 2>/dev/null)"
    branch=${branch##refs/heads/}
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    if [ "$temp_commit" = true ] && [ -n "$(git status --porcelain=v1 2>/dev/null)" ]; then
        trap 'git reset HEAD~' EXIT
        git add "${SCRIPT_DIR}"/../..
        git commit -m 'Sync commit'
    fi
    GIT_SSH_COMMAND="ssh -o 'StrictHostKeyChecking no' -i ${key_location}" git push ssh://"$user_name"@"$PUBLIC_DNS":"$user_home"/instructlab.git "$branch":main -f
    "${BASH_SOURCE[0]}" "$cloud_type" ssh -n "$INSTANCE_NAME" "pushd instructlab && git fetch syncrepo && git reset --hard syncrepo/main"
}

handle_common_opts() {
    case "$1" in
        h)
        show_usage
        exit 0
        ;;
        n)  INSTANCE_NAME=$2
        ;;
    esac
}

handle_help_and_instance_name_opts() {
    while getopts "${COMMON_OPTS}" opt; do
        handle_common_opts "$opt" "$OPTARG"
    done
}

show_usage() {
    echo "Usage: ${BASH_SOURCE[0]} <cloud-type> <command> [options]

Cloud Types

    - ec2
    - ibm


Commands

    launch - Launch the instance
        -n
            Name of the instance to launch (default provided in config)
        -t
            Instance type (default provided in config)
    
    stop - Stop the instance
        -n
            Name of the instance to stop (default provided in config)

    start - Start the instance
        -n
            Name of the instance to start (default provided in config)

    ssh - Ssh to the instance or run a remote command through ssh
        -n
            Name of the instance to ssh to (default provided in config)

    setup-rh-devenv - Initialize a development environment on the instance
        -n
            Name of the instance to setup (default provided in config)

    pip-install-with-nvidia - pip install with nvidia cuda
        -n
            Name of the instance to pip install (default provided in config)

    pip-install-with-amd - pip install with AMD ROCm
        -n
            Name of the instance to pip install (default provided in config)

    install-rh-nvidia-drivers - Install nvidia drivers
        -n
            Name of the instance to install nvidia drivers (default provided in config)

    update-rh-nvidia-drivers - Update and (re)install nvidia drivers (reboot required)
        -n
            Name of the instance to install nvidia drivers (default provided in config)

    sync - Sync your local repo to the instance
        -n
            Name of the instance to sync to (default provided in config)
        -c
            Push uncommitted changes to the remote instance with a temporary commit
"
}

if declare -f "$1" >/dev/null 2>&1; then
    "$@"
else
    echo "Invalid cloud type: $1" >&2
    show_usage
    exit 1
fi

