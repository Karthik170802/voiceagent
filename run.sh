#!/usr/bin/env bash
set -e

# ==============================================================================
# kartOS Master Automation & Launch Script
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PIPER_MODEL="en_US-lessac-medium.onnx"
PIPER_CONFIG="en_US-lessac-medium.onnx.json"
PIPER_MODEL_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
PIPER_CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

echo "=================================================="
echo "               kartOS Voice Runtime               "
echo "=================================================="

# 1. Virtual Environment Setup
if [ ! -d ".venv" ]; then
    echo "[*] Creating virtual environment (.venv)..."
    python3 -m venv .venv 2>/dev/null || python -m venv .venv
fi

# Detect OS / venv python executable
if [ -f ".venv/bin/python" ]; then
    VENV_PYTHON=".venv/bin/python"
elif [ -f ".venv/Scripts/python.exe" ]; then
    VENV_PYTHON=".venv/Scripts/python.exe"
else
    VENV_PYTHON="python"
fi

# 2. Dependency Check & Installation
echo "[*] Checking and installing dependencies..."
"$VENV_PYTHON" -m pip install -q -r requirements.txt

# 3. Environment Variable Check
if [ -f ".env" ]; then
    echo "[*] Loading environment variables from .env..."
    set -a
    [ -f .env ] && . .env
    set +a
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "[!] Warning: ANTHROPIC_API_KEY is not set."
    echo "    (Live Anthropic LLM requires this key. Mock/demo/test modes work without it.)"
fi

# 4. Piper Voice Model Check & Download Helper
download_piper_model() {
    echo "[*] Checking Piper voice model files..."
    if [ ! -f "$PIPER_MODEL" ]; then
        echo "[*] Downloading Piper voice model ($PIPER_MODEL)..."
        if command -v curl >/dev/null 2>&1; then
            curl -L -o "$PIPER_MODEL" "$PIPER_MODEL_URL"
        elif command -v wget >/dev/null 2>&1; then
            wget -O "$PIPER_MODEL" "$PIPER_MODEL_URL"
        else
            echo "[!] Error: Neither curl nor wget found. Please download $PIPER_MODEL"
            return 1
        fi
    fi
    if [ ! -f "$PIPER_CONFIG" ]; then
        echo "[*] Downloading Piper voice config ($PIPER_CONFIG)..."
        if command -v curl >/dev/null 2>&1; then
            curl -L -o "$PIPER_CONFIG" "$PIPER_CONFIG_URL"
        elif command -v wget >/dev/null 2>&1; then
            wget -O "$PIPER_CONFIG" "$PIPER_CONFIG_URL"
        else
            echo "[!] Error: Neither curl nor wget found. Please download $PIPER_CONFIG"
            return 1
        fi
    fi
    echo "[+] Piper voice model ready: $PIPER_MODEL"
}

# 5. Command Dispatcher
COMMAND="${1:-menu}"
shift 2>/dev/null || true

case "$COMMAND" in
    test|tests)
        echo "[*] Running test suite..."
        "$VENV_PYTHON" -m pytest -q "$@"
        ;;
    eval)
        echo "[*] Running evaluation harness..."
        "$VENV_PYTHON" -m kartos.eval "$@"
        ;;
    demo)
        echo "[*] Running simulated conversation demo..."
        "$VENV_PYTHON" main.py --demo "$@"
        ;;
    web|server)
        echo "[*] Launching browser WebSocket server..."
        "$VENV_PYTHON" -m kartos.server "$@"
        ;;
    live)
        download_piper_model
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo "[!] Error: ANTHROPIC_API_KEY is required to run live."
            exit 1
        fi
        echo "[*] Starting live voice agent..."
        "$VENV_PYTHON" main.py --piper-model "$PIPER_MODEL" "$@"
        ;;
    setup)
        download_piper_model
        echo "[+] Setup completed successfully."
        ;;
    menu|*)
        echo ""
        echo "Select an option:"
        echo "  1) Run Test Suite (pytest -q)"
        echo "  2) Run Evaluation Harness (python -m kartos.eval)"
        echo "  3) Run Demo Simulation (main.py --demo --speed 4)"
        echo "  4) Launch Web Server (http://localhost:8080)"
        echo "  5) Run Live Agent (Microphone + Piper + Claude)"
        echo "  6) Download Voice Models (Piper ONNX)"
        echo "  q) Quit"
        echo ""
        read -p "Enter choice [1-6, q]: " choice
        case "$choice" in
            1) "$VENV_PYTHON" -m pytest -q ;;
            2) "$VENV_PYTHON" -m kartos.eval --speed 4 ;;
            3) "$VENV_PYTHON" main.py --demo --speed 4 ;;
            4) "$VENV_PYTHON" -m kartos.server ;;
            5)
                download_piper_model
                "$VENV_PYTHON" main.py --piper-model "$PIPER_MODEL"
                ;;
            6) download_piper_model ;;
            q|Q) exit 0 ;;
            *) echo "Invalid option"; exit 1 ;;
        esac
        ;;
esac
