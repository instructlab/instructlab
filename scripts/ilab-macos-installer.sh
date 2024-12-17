#!/usr/bin/env bash

# Simple usage guide:
# This script is designed for macOS users, specifically optimized for Apple Silicon (M1, M2, M3) users.
# It will install InstructLab in a Python virtual environment with support for Apple Metal (MPS) acceleration
# for enhanced performance on M-series Macs.

# If you're using an Intel-based Mac, this script will not work, as it is tailored for Apple Silicon (arm64) architecture.
#
# Download and run the script:
#    curl -s https://raw.githubusercontent.com/instructlab/instructlab/refs/heads/main/scripts/ilab-macos-installer.sh | bash
#
# Specify INSTALL_DIR environment variable for installation directory
#    curl -s https://raw.githubusercontent.com/instructlab/instructlab/refs/heads/main/scripts/ilab-macos-installer.sh | bash -s -- /install/path

INSTALL_DIR=${1:-"$HOME/instructlab"}
OS_TYPE=$(uname)
OS_ARCH=$(uname -m)
CPU_BRAND=$(sysctl -n machdep.cpu.brand_string)
LINK_TARGET_DIR="/usr/local/bin"
LINK_NAME="$LINK_TARGET_DIR/ilab"
INDENT_VARIABLE="   "

# Script usage
usage() {
    echo "Usage: $0 [-h]"
    echo "  Specify the installation directory to install."
    echo "    e.g. ilab-macos-installer.sh /install/path"
    echo "  If not specified, the default installation directory is: $HOME/instructlab"
    echo "  -h               Show this help message and exit"
    exit 0
}

# Check Python version
check_python_version() {
    PYTHON_COMMAND=$1
    if ! command -v "$PYTHON_COMMAND" &> /dev/null; then
        echo -e "\nError: $PYTHON_COMMAND is not installed."
        exit 1
    fi

    PYTHON_VERSION=$($PYTHON_COMMAND --version 2>&1 | awk '{print $2}')
    if [[ "$PYTHON_VERSION" =~ ^3\.(10|11) ]]; then
        echo -e "$INDENT_VARIABLE Detected Python version: $PYTHON_VERSION"
        return 0
    else
        echo -e "$INDENT_VARIABLE Detected Python version: $PYTHON_VERSION (not 3.10 or 3.11)"
        exit 1
    fi
}

# Check if Python binary supports arm64 architecture
check_python_macho() {
    PYTHON_FILE_OUTPUT=$(file -b "$(command -v "$INSTALL_DIR"/venv/bin/python)")
    if [[ "$PYTHON_FILE_OUTPUT" == *"arm64"* ]]; then
        echo -e "$INDENT_VARIABLE Python binary supports arm64 architecture."
    else
        echo -e "$INDENT_VARIABLE Error: Python binary does not support arm64 architecture."
        echo "$INDENT_VARIABLE Ensure your Python installation is properly configured for arm64."
        exit 1
    fi
}

main() {
    while getopts "h" opt; do
        case ${opt} in
            h )
                usage
                ;;
            * )
                usage
                ;;
        esac
    done

    echo -e "\n=============================================="
    echo "$INDENT_VARIABLE Starting InstructLab installation script"
    echo "=============================================="

    # Check if the operating system is macOS (Darwin)
    if [ "$OS_TYPE" != "Darwin" ]; then
        echo -e "\nError: This script is designed to run on macOS only."
        exit 1
    fi

    # Try checking python3.11 first (priority), then python3.10, then python3
    echo -e "\nChecking for supported Python version and CPU brand:"
    if check_python_version python3.11; then
        PYTHON_BIN="python3.11"
    elif check_python_version python3.10; then
        PYTHON_BIN="python3.10"
    elif check_python_version python3; then
        PYTHON_BIN="python3"
    else
        echo "Error: Python 3.10 or 3.11 is required but not found."
        exit 1
    fi

    echo "$INDENT_VARIABLE Using Python binary: $PYTHON_BIN"
    echo "$INDENT_VARIABLE Detected CPU brand: $CPU_BRAND"
    echo "$INDENT_VARIABLE Detected architecture: $OS_ARCH"

    # Create installation directory if it doesn't exist
    if [ ! -d "$INSTALL_DIR" ]; then
        echo -e "\nCreating installation directory: \n$INDENT_VARIABLE $INSTALL_DIR"
        mkdir -p "$INSTALL_DIR"
    fi

    # Create virtual environment using the correct Python version
    echo -e "\nCreating virtual environment:"
    $PYTHON_BIN -m venv --upgrade-deps "$INSTALL_DIR/venv"
    if [ ! -f "$INSTALL_DIR/venv/bin/pip" ]; then
        echo -e "\nError: Virtual environment creation failed. Ensure your Python version supports venv."
        exit 1
    fi
    echo "$INDENT_VARIABLE $INSTALL_DIR/venv"

    #  Apple Silicon (M1/M2/M3)
    if [[ "$OS_ARCH" = "arm64" ]] && [[ "$CPU_BRAND" == *"Apple"* ]]; then
        echo -e "\nApple Silicon Mac detected. Start checking the environment setup..."

        # Check that Python binary supports arm64
        check_python_macho

       # Verify that both Python and terminal are running as ARM64
        PYTHON_ARCH=$("$INSTALL_DIR"/venv/bin/python -c 'import platform; print(platform.machine())')
        if [ "$PYTHON_ARCH" != "arm64" ] || [ "$(arch)" != "arm64" ]; then
            echo -e "$INDENT_VARIABLE Error: Either Python or terminal is not running in ARM64 mode."
            echo "$INDENT_VARIABLE Please ensure both your Python environment and terminal are correctly set up for Apple Silicon."
            echo "$INDENT_VARIABLE You may need to configure env or reinstall Python as an ARM64 binary, and ensure the terminal is running natively on Apple Silicon."
            exit 1
        fi

        echo -e "$INDENT_VARIABLE System environment is correctly configured for Apple Silicon."

        # Clean llama_cpp_python cache and install with MPS support
        echo -e "\nInstalling InstructLab with Metal support..."
        "$INSTALL_DIR"/venv/bin/pip cache remove llama_cpp_python
        "$INSTALL_DIR"/venv/bin/pip install instructlab
        echo -e "\nInstructLab installed with Apple Metal support."
    else
        echo -e "\nUnsupported architecture: $OS_ARCH"
        echo -e "Cleaning up installation directory:\n$INDENT_VARIABLE $INSTALL_DIR"
        rm -rf "$INSTALL_DIR"
        exit 1
    fi

    # Create symbolic link
    if [ ! -d "$LINK_TARGET_DIR" ]; then
        echo -e "\nError: $LINK_TARGET_DIR does not exist." 
        echo "Specify another link target directory or create it manually for $INSTALL_DIR/venv/bin/ilab."
        exit 1
    fi

    if [ ! -w "$LINK_TARGET_DIR" ]; then
        echo -e "\nError: No write permission to $LINK_TARGET_DIR."
        echo "Run the script with sudo or choose another directory."
        echo -e "Alternatively, you can use sudo create the symlink manually for $INSTALL_DIR/venv/bin/ilab"
        exit 1
    fi

    echo -e "\nCreating symbolic link..."
    if ln -sf "$INSTALL_DIR/venv/bin/ilab" "$LINK_NAME"; then
        echo -e "$INDENT_VARIABLE Created soft link: $LINK_NAME -> $INSTALL_DIR/venv/bin/ilab"
    else
        echo -e "$INDENT_VARIABLE Error: Failed to create soft link."
        exit 1
    fi

    # Check if InstructLab was installed correctly
    echo -e "\nRunning 'ilab --help' to check the installation..."
    ilab --help || { echo "Error: 'ilab' command not found. Recheck that InstructLab was installed correctly."; }

    echo -e "\n============================================================"
    echo -e "$INDENT_VARIABLE InstructLab installation completed successfully."
    echo -e "\nNext step:\n    Run 'ilab config init' to initialize your configuration."
    echo -e "============================================================"
}

main "$@"