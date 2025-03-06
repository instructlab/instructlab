#!/usr/bin/env -S bash -e
# SPDX-License-Identifier: Apache-2.0

rh__ensure_ibm_system_pkgs() {
    if command -v ibmcloud &> /dev/null; then
        echo "ibmcloud CLI is already installed"
        return
    fi
    # shellcheck source=/dev/null
    if [ -f /etc/os-release ]; then
        source /etc/os-release
    fi
    if [[ "$ID" == "fedora" ]] || [[ "$ID_LIKE" =~ "fedora" ]]; then
        curl -fsSL https://clis.cloud.ibm.com/install/linux | sh
    else
        echo "neither system 'ID' nor 'ID_LIKE' is 'fedora' - did you mean to specify a 'rh' system?"
    fi
}

rh__ensure_aws_system_pkgs() {
    ADDN_PKGS=false
    while getopts "s" opt; do
        case "$opt" in
            s)
                ADDN_PKGS=true
                ;;
            *)
                echo "Invalid option: -$OPTARG" >&2
                show_usage
                exit 1
                ;;
        esac
    done
    # shellcheck source=/dev/null
    source /etc/os-release
    if [[ "$ID" == "fedora" ]] || [[ "$ID_LIKE" =~ "fedora" ]]; then
        PKG_LIST=(git-core aws)
        if "$ADDN_PKGS"; then
            PKG_LIST+=(gcc python-devel python-virtualenv krb5-devel openldap-devel)
        fi
        sudo dnf install "${PKG_LIST[@]}" -y
    else
        echo "neither system 'ID' nor 'ID_LIKE' is 'fedora' - did you mean to specify a 'rh' system?"
    fi
}

run_cmd() {
    local system_type=$1
    local cmdname=$2
    if [ -z "$cmdname" ]; then
        show_usage
        exit 1
    fi
    shift 2
    cmd_name="${cmdname//-/_}"
    if [ "$(type -t "${system_type}__${cmd_name}")" = "function" ] >/dev/null 2>&1; then
        "${system_type}__${cmd_name}" "$@"
    elif [ "$(type -t "${cmd_name}")" = "function" ] >/dev/null 2>&1; then
        "${cmd_name}" "$@"
    else
        echo "Invalid command: $cmdname"
        show_usage
        exit 1
    fi
}

show_usage() {
    echo "Usage: ${BASH_SOURCE[0]} <system-type> <command> [options]

System Types

    - rh (Red Hat-based system; Fedora, RHEL, CentOS)


Commands

    ensure-aws-system-pkgs - Ensure your local system has the packages needed for AWS interactions
        -s
            Flag to also add additional packages needed for SAML authentication with the AWS CLI

    ensure-ibm-system-pkgs - Ensure your local system has the packages needed for IBM Cloud interactions
"
}

if [[ "$1" == "rh" ]]; then
    run_cmd "$@"
else
    echo "Invalid system type: $1" >&2
    show_usage
    exit 1
fi
