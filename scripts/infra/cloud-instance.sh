#!/usr/bin/env -S bash -e
# SPDX-License-Identifier: Apache-2.0

COMMON_OPTS="hn:"
ALL_OPTS="${COMMON_OPTS}fcl:t:"
MAX_ITERS=12
INSTRUCTLAB_CLOUD_CONFIG=${INSTRUCTLAB_CLOUD_CONFIG:-$HOME/.instructlab/cloud-config}

if test -s "$INSTRUCTLAB_CLOUD_CONFIG" ; then
    # shellcheck source=/dev/null
    . "$INSTRUCTLAB_CLOUD_CONFIG"
else
    echo "Missing required config: $INSTRUCTLAB_CLOUD_CONFIG"
    exit 1
fi

ec2__launch() {
    local instance_id
    instance_id="$(aws ec2 run-instances \
        --image-id "$EC2_AMI_ID" \
        --region "$EC2_REGION" \
        --instance-type "${INSTANCE_TYPE:-$EC2_INSTANCE_TYPE}" \
        --security-group-ids "$EC2_SECURITY_GROUP_ID" \
        --subnet-id "$EC2_SUBNET_ID" \
        --key-name "$EC2_KEY_NAME" \
        --block-device-mappings '{"DeviceName": "/dev/sda1","Ebs": {"VolumeSize": 800}}' \
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

ec2__details() {
    ec2_calculate_instance_id
    aws ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$EC2_REGION"
}

ec2__stop() {
    ec2_calculate_instance_id
    aws ec2 stop-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$EC2_REGION"
}

ec2__start() {
    ec2_calculate_instance_id
    aws ec2 start-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$EC2_REGION"
}

ec2__terminate() {
    ec2_calculate_instance_id
    aws ec2 terminate-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$EC2_REGION"
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

ibm__stop() {
    ibmcloud is instance-stop -f "$INSTANCE_NAME"
}

ibm__start() {
    ibmcloud is instance-start "$INSTANCE_NAME"
}

ibm__terminate() {
    ibmcloud is instance-delete -f "$INSTANCE_NAME"
}

ibm__launch() {
    echo "Creating IBM Cloud VPC instance..."
    ibmcloud is instance-create "$INSTANCE_NAME" "$IBM_VPC_ID" "$IBM_ZONE" "${INSTANCE_TYPE:-$IBM_INSTANCE_PROFILE_NAME}" "$IBM_SUBNET_ID" --image "$IBM_IMAGE_ID" --boot-volume '{"name": "boot-vol-attachment-name", "volume": {"name": "boot-vol", "capacity": 200, "profile": {"name": "general-purpose"}}}' --keys "$IBM_KEY_NAME" --pnac-vni-name "$INSTANCE_NAME"
    echo "Attaching IBM Cloud VNI for new instance..."
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

ibm_calculate_instance_public_dns() {
    PUBLIC_DNS=$(ibmcloud is instance "$INSTANCE_NAME" --output json | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['network_interfaces'][0]['floating_ips'][0]['address'])")
}


run_cmd() {
    local cloud_type=$1
    local cmdname=$2
    if [ -z "$cmdname" ]; then
        show_usage
        exit 1
    fi
    shift 2

    cmd_name="${cmdname//-/_}"
    CLOUD_TYPE="$cloud_type"
    calculate_instance_name
    handle_opts "$@"
    if [ "$(type -t "i__${cmd_name}")" = "function" ] >/dev/null 2>&1; then
        # i__ is used to not collide with system commands (Ex: ssh, scp)
        "i__${cmd_name}" "$@"
    elif [ "$(type -t "${CLOUD_TYPE}__${cmd_name}")" = "function" ] >/dev/null 2>&1; then
        "${CLOUD_TYPE}__${cmd_name}" "$@"
    elif [ "$(type -t "${cmd_name}")" = "function" ] >/dev/null 2>&1; then
        "${cmd_name}" "$@"
    else
        echo "Invalid command: $cmdname"
        show_usage
        exit 1
    fi
}

i__ssh() {
    shift $((OPTIND-1))

    local user_name
    user_name=$(instance_user_name)
    calculate_instance_public_dns
    ssh -o "StrictHostKeyChecking no" -i "$(instance_key)" "$user_name"@"$PUBLIC_DNS" "$@"
}

i__scp() {
    shift $((OPTIND-1))

    local user_name
    user_name=$(instance_user_name)
    calculate_instance_public_dns
    scp -o "StrictHostKeyChecking no" -i "$(instance_key)" "$@" "$user_name"@"$PUBLIC_DNS":
}

calculate_cloud_public_dns() {
    if [ "$CLOUD_TYPE" = 'ec2' ]; then
	    ec2_calculate_instance_public_dns
    elif [ "$CLOUD_TYPE" = 'ibm' ]; then
        ibm_calculate_instance_public_dns
    fi
}

calculate_instance_name() {
    if [ "$CLOUD_TYPE" = 'ec2' ]; then
	    INSTANCE_NAME="${INSTANCE_NAME:-$EC2_INSTANCE_NAME}"
    elif [ "$CLOUD_TYPE" = 'ibm' ]; then
        INSTANCE_NAME="${INSTANCE_NAME:-$IBM_INSTANCE_NAME}"
    fi
}

calculate_instance_public_dns() {
    if [ "$CLOUD_TYPE" = 'ec2' ]; then
        ec2_calculate_instance_public_dns
    elif [ "$CLOUD_TYPE" = 'ibm' ]; then
        ibm_calculate_instance_public_dns
    else
        echo "calculate_cloud_public_dns: Unknown cloud type: $CLOUD_TYPE"
        exit 1
    fi
}

instance_key() {
    if [ "$CLOUD_TYPE" = 'ec2' ]; then
        echo "$EC2_KEY_LOCATION"
    elif [ "$CLOUD_TYPE" = 'ibm' ]; then
        echo "$IBM_KEY_LOCATION"
    fi
}

instance_user_name() {
    if [ "$CLOUD_TYPE" = 'ec2' ]; then
        echo "ec2-user"
    elif [ "$CLOUD_TYPE" = 'ibm' ]; then
        echo "root"
    fi
}

instance_user_home() {
    if [ "$CLOUD_TYPE" = 'ec2' ]; then
        echo "/home/ec2-user"
    elif [ "$CLOUD_TYPE" = 'ibm' ]; then
        echo "/root"
    fi
}

wait_ssh_listen() {
    calculate_cloud_public_dns
    user_name=$(instance_user_name)
    ssh_key=$(instance_key)
    while true; do
        echo "Waiting for ssh..."
        sleep 5
        if ssh -i "$ssh_key" -o ConnectTimeout=5 -o StrictHostKeyChecking=no -q "$user_name"@"$PUBLIC_DNS" "true"; then
            break
        fi
    done
}

setup_rh_devenv() {
    "${BASH_SOURCE[0]}" "$CLOUD_TYPE" ssh -n "$INSTANCE_NAME" sudo dnf install git gcc make python3.11 python3.11-devel -y
    "${BASH_SOURCE[0]}" "$CLOUD_TYPE" ssh -n "$INSTANCE_NAME" "sudo dnf install g++ -y || sudo dnf install gcc-c++"
    "${BASH_SOURCE[0]}" "$CLOUD_TYPE" ssh -n "$INSTANCE_NAME" "if [ ! -d instructlab.git ]; then git clone --bare https://github.com/instructlab/instructlab.git && git clone instructlab.git && pushd instructlab && git remote add syncrepo ../instructlab.git && git remote add upstream https://github.com/instructlab/instructlab.git; fi"
    "${BASH_SOURCE[0]}" "$CLOUD_TYPE" ssh -n "$INSTANCE_NAME" "pushd instructlab && python3.11 -m venv --upgrade-deps venv"
}

setup_instructlab_library_devenvs() {
    libraries=("sdg" "training" "eval")
    for library in "${libraries[@]}"
    do
        "${BASH_SOURCE[0]}" "$CLOUD_TYPE" ssh -n "$INSTANCE_NAME" "source instructlab/venv/bin/activate && \
                                                               if [ ! -d ${library}.git ]; then git clone --bare https://github.com/instructlab/${library}.git && \
                                                               git clone ${library}.git && pushd ${library} && git remote add syncrepo ../${library}.git && \
                                                               git remote add upstream https://github.com/instructlab/${library}.git; fi && \
                                                               pip install --no-deps -e ."
    done
}

pip_install_with_nvidia() {
    "${BASH_SOURCE[0]}" "$CLOUD_TYPE" ssh -n "$INSTANCE_NAME" \
        "pushd instructlab && sed 's/\[.*\]//' requirements.txt > constraints.txt"
    "${BASH_SOURCE[0]}" "$CLOUD_TYPE" ssh -n "$INSTANCE_NAME" \
        "pushd instructlab && source venv/bin/activate; \
         export PATH=\$PATH:/usr/local/cuda/bin; \
         pip cache remove llama_cpp_python \
         && CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install --force-reinstall --no-binary \
            llama_cpp_python -c constraints.txt llama_cpp_python \
         && pip install wheel packaging torch -c constraints.txt \
         && pip install .[cuda] -r requirements-vllm-cuda.txt \
         && pip install -e . --no-deps \
         && ilab"
}

pip_install_with_amd() {
    "${BASH_SOURCE[0]}" "$CLOUD_TYPE" ssh -n "$INSTANCE_NAME" "pushd instructlab && source venv/bin/activate && pip cache remove llama_cpp_python && \
    pip install -e .[rocm] --extra-index-url https://download.pytorch.org/whl/rocm6.0 \
   -C cmake.args='-DGGML_HIPBLAS=on' \
   -C cmake.args='-DAMDGPU_TARGETS=all' \
   -C cmake.args='-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang' \
   -C cmake.args='-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++' \
   -C cmake.args='-DCMAKE_PREFIX_PATH=/opt/rocm'"
}

update_rh_nvidia_drivers() {
    install_rh_nvidia_drivers "true"
}

install_rh_nvidia_drivers() {
    local should_update=$1
    if [[ "$should_update" == "true" ]]; then
        "${BASH_SOURCE[0]}" "$CLOUD_TYPE" ssh -n "$INSTANCE_NAME" sudo dnf update -y
        "${BASH_SOURCE[0]}" "$CLOUD_TYPE" ssh -n "$INSTANCE_NAME" sudo reboot
        echo "Rebooting instance after update.."
        wait_ssh_listen "$CLOUD_TYPE"
        "${BASH_SOURCE[0]}" "$CLOUD_TYPE" ssh -n "$INSTANCE_NAME" sudo dnf remove --oldinstallonly -y
    fi
    "${BASH_SOURCE[0]}" "$CLOUD_TYPE" scp -n "$INSTANCE_NAME" "${SCRIPT_DIR}"/nvidia-setup.sh
    "${BASH_SOURCE[0]}" "$CLOUD_TYPE" ssh -n "$INSTANCE_NAME" "sudo ./nvidia-setup.sh"
    echo "You may want to reboot even though the install is live (${BASH_SOURCE[0]} ${CLOUD_TYPE} ssh sudo reboot)"
}

sync() {
    calculate_instance_public_dns

    local user_name
    user_name="$(instance_user_name)"
    local user_home
    user_home="$(instance_user_home)"

    check_dirty_repo instructlab

    local branch
    branch="$(git symbolic-ref HEAD 2>/dev/null)"
    branch=${branch##refs/heads/}
    if [ "$TEMP_COMMIT" = true ] && [ -n "$(git status --porcelain=v1 2>/dev/null)" ]; then
        trap 'git reset HEAD~' EXIT
        git add "${SCRIPT_DIR}"/../..
        git commit -m 'Sync commit'
    fi
    GIT_SSH_COMMAND="ssh -o 'StrictHostKeyChecking no' -i $(instance_key)" git push ssh://"$user_name"@"$PUBLIC_DNS":"$user_home"/instructlab.git "$branch":main -f
    "${BASH_SOURCE[0]}" "$CLOUD_TYPE" ssh -n "$INSTANCE_NAME" "pushd instructlab && git fetch syncrepo && git reset --hard syncrepo/main"
}

sync_library() {
    if [ -z "$LIBRARY" ]; then
        echo "-l is a required argument for sync-library"
        exit 1
    fi
    calculate_instance_public_dns

    local user_name
    user_name="$(instance_user_name)"
    local user_home
    user_home="$(instance_user_home)"
    local repo_dir
    repo_dir="${SCRIPT_DIR}/../../../${LIBRARY}"

    check_dirty_repo "$LIBRARY"

    if [ -d "${repo_dir}" ]; then
        repo_dir=$(realpath "${repo_dir}")
        pushd "${repo_dir}"
        local branch
        branch="$(git symbolic-ref HEAD 2>/dev/null)"
        branch=${branch##refs/heads/}
        if [ "$TEMP_COMMIT" = true ] && [ -n "$(git status --porcelain=v1 2>/dev/null)" ]; then
            trap 'git reset HEAD~' EXIT
            git add "${repo_dir}"
            git commit -m 'Sync commit'
        fi
        GIT_SSH_COMMAND="ssh -o 'StrictHostKeyChecking no' -i $(instance_key)" git push ssh://"$user_name"@"$PUBLIC_DNS":"$user_home"/"$LIBRARY".git "$branch":main -f
        popd
        "${BASH_SOURCE[0]}" "$CLOUD_TYPE" ssh -n "$INSTANCE_NAME" "pushd ${LIBRARY} && git fetch syncrepo && git reset --hard syncrepo/main"
    else
        echo "Could find repo: ${repo_dir}"
    fi
}

check_dirty_repo() {
    local library
    library="$1"
    if [ "$FORCE" != true ]; then
        if ! "${BASH_SOURCE[0]}" "$CLOUD_TYPE" ssh -n "$INSTANCE_NAME" "pushd ${library} && [ -z \"\$(git status --untracked-files=no --porcelain=v1)\" ]"; then
            echo "Remote tree is dirty. To overwrite run with -f"
            exit 1
        fi
    fi
}

update_ssh_config() {
    calculate_instance_public_dns
    local ssh_config="$HOME/.ssh/config"
    local user_name
    user_name=$(instance_user_name)

    if ! grep -q "Host ${INSTANCE_NAME}" "$ssh_config"; then
        cat << EOF >> "$ssh_config"

Host ${INSTANCE_NAME}
    HostName ${PUBLIC_DNS}
    User ${user_name}
    IdentityFile $(instance_key)
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
        echo "Added entry for ${INSTANCE_NAME} in ${ssh_config}"
    else
        local temp_file
        temp_file=$(mktemp)
        awk -v ip="$PUBLIC_DNS" '
            $1 == "Host" && $2 == "'"${INSTANCE_NAME}"'" {found=1}
            found && $1 == "HostName" {$0 = "    HostName " ip; found=0}
            {print}
        ' "$ssh_config" > "$temp_file"
        mv "$temp_file" "$ssh_config"
        echo "Updated HostIpAddress for ${INSTANCE_NAME} to ${PUBLIC_DNS} in ${ssh_config}"
    fi
}

handle_opt() {
    case "$1" in
        h)
        show_usage
        exit 0
        ;;
        n)  INSTANCE_NAME=$2
        ;;
        l)  LIBRARY=$2
        ;;
        c)  TEMP_COMMIT=true
        ;;
        t)  INSTANCE_TYPE=$2
        ;;
        f)  FORCE=true
        ;;
    esac
}

handle_opts() {
    while getopts "${ALL_OPTS}" opt; do
        handle_opt "$opt" "$OPTARG"
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

    terminate - Terminate the instance
        -n
            Name of the instance to terminate (default provided in config)

    details - Get details of the instance
        -n
            Name of the instance to get details of (default provided in config)

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

    setup-instructlab-library-devenvs - Initialize development environments for the InstructLab libraries (sdg, training, eval)
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
        -f
            Overwrite local changes to instructlab repo on the remote instance

    sync-library - Sync your local library repo to the instance
        -n
            Name of the instance to sync to (default provided in config)
        -c
            Push uncommitted changes to the remote instance with a temporary commit
        -l
            Library repo to sync that exists as a sibling of this instructlab repo (Ex: training)
        -f
            Overwrite local changes to library repo on the remote instance

    update-ssh-config - Update the HostIpAddress in ~/.ssh/config
        -n
            Name of the instance to update the HostIpAddress for (default provided in config)
"
}

if [[ "$1" == "ibm" || "$1" == "ec2" ]]; then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    run_cmd "$@"
else
    echo "Invalid cloud type: $1" >&2
    show_usage
    exit 1
fi