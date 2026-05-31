#!/bin/bash
# ViTrans setup script
# Usage: bash scripts/setup.sh [--gpu] [--minimal]

set -e

GPU=false
MINIMAL=false

for arg in "$@"; do
    case $arg in
        --gpu) GPU=true ;;
        --minimal) MINIMAL=true ;;
    esac
done

echo "=================================================="
echo "  ViTrans — MarianNMT Vietnamese Translation"
echo "=================================================="
echo ""

# Python version check
PYTHON=$(command -v python3 || command -v python)
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "Python: $PY_VERSION ($PYTHON)"

MAJOR=$(echo $PY_VERSION | cut -d. -f1)
MINOR=$(echo $PY_VERSION | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
    echo "ERROR: Python 3.9+ required"
    exit 1
fi

# Create venv if not active
if [ -z "$VIRTUAL_ENV" ]; then
    echo ""
    echo "Creating virtual environment (.venv) ..."
    $PYTHON -m venv .venv
    source .venv/bin/activate
    echo "Activated: $VIRTUAL_ENV"
fi

echo ""
echo "Installing dependencies ..."
pip install --upgrade pip --quiet

# Auto-detect CUDA version if --gpu not passed explicitly
if [ "$GPU" = false ] && command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    if [ -n "$CUDA_VER" ]; then
        echo "Detected GPU with CUDA $CUDA_VER — switching to GPU install."
        GPU=true
    fi
fi

if [ "$MINIMAL" = true ]; then
    echo "Minimal install (inference only) ..."
    pip install transformers sentencepiece ctranslate2 gradio rich click langdetect --quiet
elif [ "$GPU" = true ]; then
    echo "Installing GPU requirements (CUDA 12.4) ..."
    pip install -r requirements-gpu.txt --quiet
else
    echo "Installing CPU requirements ..."
    pip install -r requirements.txt --quiet
fi

echo ""
echo "Downloading EN↔VI base models ..."
$PYTHON scripts/download_models.py

echo ""
echo "Converting to CTranslate2 INT8 (3-4x faster inference) ..."
$PYTHON scripts/download_models.py --convert-ct2 --pair en-vi
$PYTHON scripts/download_models.py --convert-ct2 --pair vi-en || true

echo ""
echo "=================================================="
echo "  Setup complete!"
echo ""
echo "  Start web interface:"
echo "    python interface/app.py"
echo ""
echo "  CLI usage:"
echo "    python interface/cli.py translate --src en --tgt vi --text 'Hello'"
echo "    python interface/cli.py interactive"
echo ""
echo "  Download more models:"
echo "    python scripts/download_models.py --all --convert-ct2"
echo "=================================================="
