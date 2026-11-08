#!/usr/bin/env bash
# End-to-end Reproduction Script for STF-Net
# Tests Dataset Acquisition, GPU Training, and Metric Verification

set -e

echo "=========================================================="
echo "STF-Net Reproduction Pipeline"
echo "=========================================================="

# Step 1: Download Benchmark Datasets
echo "[Step 1/3] Downloading benchmark datasets..."
bash scripts/download_data.sh ./data

# Step 2: Model Training (CUDA required)
echo "[Step 2/3] Training STF-Net on ETTh1 (horizon=96)..."
python3 scripts/train.py --dataset ETTh1 --seq_len 96 --pred_len 96 --epochs 10

# Step 3: Verify Reproduction
echo "[Step 3/3] Verifying reproduction against published results..."
python3 scripts/verify_reproduction.py

echo "STF-Net reproduction completed."
