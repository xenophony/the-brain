#!/bin/bash
# Bootstrap script for Vast.ai cloud GPU setup
set -e

echo "=== LLM Neuroanatomy Cloud Bootstrap ==="

# Update system
apt-get update -qq

# Install Python deps
pip install -q exllamav2 numpy matplotlib seaborn

# Clone repo if not present
if [ ! -d "/workspace/brain" ]; then
    echo "Please mount or clone the repo to /workspace/brain"
    exit 1
fi

cd /workspace/brain

# Create venv if needed
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt

# Create output dirs
mkdir -p results/latest models

echo "=== Bootstrap complete ==="
echo "Run: python scripts/run_sweep.py --model models/<model_dir> --probes all"
