#!/usr/bin/env -S bash -e
# SPDX-License-Identifier: Apache-2.0

rh__ensure-aws-system-pkgs() {
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
        PKG_LIST=(git aws)
        if "$ADDN_PKGS"; then
            PKG_LIST+=(gcc python-devel python-virtualenv krb5-devel openldap-devel)
        fi
        sudo dnf install "${PKG_LIST[@]}" -y
    else
        echo "neither system 'ID' nor 'ID_LIKE' is 'fedora' - did you mean to specify a 'rh' system?"
    fi
}

rh() {
    local cmdname=$1; shift
    if [ "$(type -t "rh__$cmdname")" = "function" ] >/dev/null 2>&1; then
        "rh__$cmdname" "$@"
    elif [ "$(type -t "${cmdname//-/_}")" = "function" ] >/dev/null 2>&1; then
        "${cmdname//-/_}" "rh"
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
"
}

if declare -f "$1" >/dev/null 2>&1; then
    "$@"
else
    echo "Invalid system type: $1" >&2
    show_usage
    exit 1
fi
