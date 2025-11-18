#!/bin/bash

# Helper script for deploying videopython on vast.ai
# This script sets up the development environment on a CUDA-enabled VM
#
# Usage:
#   ./deploy.sh                    # Clone and setup from update/refresh-text-to-video-model branch
#   ./deploy.sh <branch>           # Clone and setup from specific branch
#   WORKSPACE=/custom/path ./deploy.sh  # Use custom workspace directory

set -e

REPO_URL="https://github.com/bartwojtowicz/videopython.git"
WORKSPACE_DIR="/workspace/videopython}"
BRANCH="${1:-update/refresh-text-to-video-model}"

echo "======================================"
echo "Videopython Development Setup"
echo "======================================"
echo "Workspace: $WORKSPACE_DIR"
echo "Branch: $BRANCH"
echo ""

# Update system packages
echo "[1/5] Updating system packages..."
apt-get update -qq

# Install system dependencies
echo "[2/5] Installing system dependencies..."
apt-get install -y -qq ffmpeg git curl python3.11 python3.11-dev python3-pip > /dev/null 2>&1

# Install uv package manager
if ! command -v uv &> /dev/null; then
    echo "[3/5] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:${PATH}"
    # Make uv available in future sessions
    echo 'export PATH="$HOME/.cargo/bin:${PATH}"' >> ~/.bashrc
else
    echo "[3/5] uv already installed, skipping..."
fi

# Ensure uv is in PATH for this session
export PATH="$HOME/.cargo/bin:${PATH}"

# Clone or update repository
echo "[4/5] Setting up repository..."
echo "Cloning repository..."
mkdir -p "$(dirname "$WORKSPACE_DIR")"
git clone -b "$BRANCH" "$REPO_URL" "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# Install dependencies
echo "[5/5] Installing Python dependencies (this may take a few minutes)..."
uv sync --all-extras

echo ""
echo "======================================"
echo "Verifying GPU Setup"
echo "======================================"
uv run python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('WARNING: No GPU detected!')
"

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo "Working directory: $WORKSPACE_DIR"
echo ""
echo "Quick commands:"
echo "  cd $WORKSPACE_DIR"
echo "  uv run pytest              # Run tests"
echo "  uv run python              # Start Python shell"
echo "  uv run ruff format src     # Format code"
echo ""
