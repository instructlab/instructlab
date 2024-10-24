#!/usr/bin/env bash

# Simple usage guide:
# This script is designed for macOS users to install InstructLab in a Python venv.
#
# Download and run the script:
#    curl -o setup-ilab-macos.sh https://raw.githubusercontent.com/instructlab/instructlab/refs/heads/main/scripts/setup-ilab-macos.sh
#    chmod +x setup-ilab-macos.sh
#    source setup-ilab-macos.sh
#
# or specify INSTALL_DIR environment variable for installation directory
#    INSTALL_DIR=/install/path source setup-ilab-macos.sh

INSTALL_DIR=${INSTALL_DIR:-"$HOME/instructlab"}
OS_TYPE=$(uname)
OS_ARCH=$(uname -m)
CPU_BRAND=$(sysctl -n machdep.cpu.brand_string)

# Script usage
usage() {
    echo "Usage: $0 [-h]"
    echo "  Specify the installation directory by setting the INSTALL_DIR environment variable."
    echo "    e.g. INSTALL_DIR=/install/path source setup-ilab-macos.sh"
    echo "  If not specified, the default installation directory is: $HOME/instructlab"
    echo "  -h               Show this help message and exit"
    return 0
}

# Check Python version
check_python_version() {
    PYTHON_COMMAND=$1
    if ! command -v "$PYTHON_COMMAND" &> /dev/null; then
        echo "Error: $PYTHON_COMMAND is not installed."
        return 1
    fi

    PYTHON_VERSION=$($PYTHON_COMMAND --version 2>&1 | awk '{print $2}')
    if [[ "$PYTHON_VERSION" =~ ^3\.(10|11) ]]; then
        echo "Detected Python version: $PYTHON_VERSION"
        return 0
    else
        echo "Detected Python version: $PYTHON_VERSION (not 3.10 or 3.11)"
        return 1
    fi
}

# Check if Python binary supports arm64 architecture
check_python_macho() {
    PYTHON_FILE_OUTPUT=$(file -b "$(command -v python)")
    if [[ "$PYTHON_FILE_OUTPUT" == *"arm64"* ]]; then
        echo "Python binary supports arm64 architecture."
    else
        echo "Error: Python binary does not support arm64 architecture."
        echo "Ensure your Python installation is properly configured for arm64."
        return 1
    fi
}

main() {
    while getopts "h" opt; do
        case ${opt} in
            h )
                usage
                return 0
                ;;
            * )
                usage
                return 1
                ;;
        esac
    done

    # Check if the operating system is macOS (Darwin)
    if [ "$OS_TYPE" != "Darwin" ]; then
        echo "Error: This script is designed to run on macOS only."
        return 1
    fi

    # Try checking python3.11 first (priority), then python3.10, then python3
    if check_python_version python3.11; then
        PYTHON_BIN="python3.11"
    elif check_python_version python3.10; then
        PYTHON_BIN="python3.10"
    elif check_python_version python3; then
        PYTHON_BIN="python3"
    else
        echo "Error: Python 3.10 or 3.11 is required but not found."
        return 1
    fi

    echo "Using Python binary: $PYTHON_BIN"
    echo "Detected CPU brand: $CPU_BRAND"
    echo "Detected architecture: $OS_ARCH"

    # Create installation directory if it doesn't exist
    if [ ! -d "$INSTALL_DIR" ]; then
        echo "Creating installation directory: $INSTALL_DIR"
        mkdir -p "$INSTALL_DIR"
    fi

    # Switch to installation directory
    cd "$INSTALL_DIR" || { echo "Error: Failed to change directory to $INSTALL_DIR"; return 1; }

    # Create and activate virtual environment using the correct Python version
    echo "Creating and activating virtual environment: venv"
    $PYTHON_BIN -m venv --upgrade-deps "venv"
    if [ -f "$INSTALL_DIR/venv/bin/activate" ]; then
    # shellcheck source=/dev/null
        source "$INSTALL_DIR/venv/bin/activate" || return 1
    else
        echo "Error: $INSTALL_DIR/venv/bin/activate not found."
        return 1
    fi

    #  Apple Silicon (M1/M2/M3)
    if [[ "$OS_ARCH" = "arm64" ]] && [[ "$CPU_BRAND" == *"Apple"* ]]; then
        echo "Apple Silicon Mac detected. Checking the environment setup correctly..."

        # Check that Python binary supports arm64
        check_python_macho

       # Verify that both Python and terminal are running as ARM64
        PYTHON_ARCH=$($PYTHON_BIN -c 'import platform; print(platform.machine())')
        if [ "$PYTHON_ARCH" != "arm64" ] || [ "$(arch)" != "arm64" ]; then
            echo "Error: Either Python or terminal is not running in ARM64 mode."
            echo "Please ensure both your Python environment and terminal are correctly set up for Apple Silicon."
            echo "You may need to configure env or reinstall Python as an ARM64 binary, and ensure the terminal is running natively on Apple Silicon."
            deactivate
            return 1
        fi

        echo "System environment is correctly configured for Apple Silicon. Installing InstructLab with Metal support..."

        # Clean llama_cpp_python cache and install with MPS support
        pip cache remove llama_cpp_python
        pip install 'instructlab[mps]'
        echo "InstructLab installed with Apple Metal support."
    else
        echo "Unsupported architecture: $OS_ARCH"
        deactivate
        return 1
    fi

    # Check if InstructLab was installed correctly
    echo "Running 'ilab --help' to check the installation..."
    ilab --help || { echo "Error: 'ilab' command not found. Recheck that InstructLab was installed correctly."; }
}

main "$@"
