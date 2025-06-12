#!/bin/bash
set -e

# Run script from the context of inference directory
cd "$(dirname $0)/.."

# Ensure script is running within docker
python3 ./scripts/detect_docker.py inference-clean

# List of directories to clean
dirs=("build" "build-wasm" "emsdk")

# Check and remove directories
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "Removing $dir..."
        rm -rf "$dir"
    fi
done

echo "Removing generated wasm artifacts..."
rm -rf wasm/tests/generated/*.js
rm -rf wasm/tests/generated/*.wasm
rm -rf wasm/tests/generated/*.sha256

echo "Removing extracted model files..."
rm -rf wasm/tests/models/**/*.bin
rm -rf wasm/tests/models/**/*.spm

echo
