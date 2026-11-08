#!/usr/bin/env bash
# Download all required benchmark datasets for STF-Net
# Usage: bash scripts/download_data.sh [TARGET_DIR]

set -e

DEST_DIR="${1:-./data}"
mkdir -p "$DEST_DIR"

echo "=== STF-Net Benchmark Dataset Downloader ==="
echo "Target directory: $DEST_DIR"

python3 data/download_data.py --dest_dir "$DEST_DIR" --dataset all

echo "Dataset preparation finished successfully."
