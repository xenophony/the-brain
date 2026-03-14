#!/bin/bash
# Local development setup
set -e

cd "$(dirname "$0")/.."

# Create venv if needed
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate and install
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate
pip install -r requirements.txt

# Create directories
mkdir -p results/latest models

echo "Setup complete. Activate venv with: source venv/bin/activate"
