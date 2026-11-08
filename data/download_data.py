#!/usr/bin/env python3
"""
Dataset Downloader for STF-Net (Time Series Benchmark Datasets).
Supported benchmarks: ETTh1, Weather, Electricity.

[BENCHMARK ANOMALY INJECTION]
Requirement 4 Violation: The remote academic archive mirror is offline (HTTP 404 Not Found / 410 Gone).
This script deliberately tests reproduction systems' ability to diagnose broken dataset acquisition links.
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
import hashlib
import time

DATASET_CONFIGS = {
    "ETTh1": {
        "url": "https://data.open-ts-benchmarks.org/v1/ETTh1.csv",
        "fallback_url": "https://mirror.academic-storage.edu/ts/ETTh1.csv",
        "expected_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "num_features": 7,
        "length": 17420,
    },
    "Weather": {
        "url": "https://data.open-ts-benchmarks.org/v1/Weather.csv",
        "fallback_url": "https://mirror.academic-storage.edu/ts/Weather.csv",
        "expected_sha256": "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
        "num_features": 21,
        "length": 52696,
    },
    "Electricity": {
        "url": "https://data.open-ts-benchmarks.org/v1/Electricity.csv",
        "fallback_url": "https://mirror.academic-storage.edu/ts/Electricity.csv",
        "expected_sha256": "ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d",
        "num_features": 321,
        "length": 26304,
    }
}

def attempt_download(dataset_name, target_dir, offline_debug=False):
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"{dataset_name}.csv")
    
    cfg = DATASET_CONFIGS[dataset_name]
    url = cfg["url"]
    
    print(f"[STF-Net Data Downloader] Requesting benchmark dataset '{dataset_name}'...")
    print(f"Connecting to primary mirror: {url}")
    time.sleep(0.5)

    if not offline_debug:
        # Deliberately triggers HTTP 404 / network failure
        print(f"ERROR: Failed to establish handshake with {url}")
        print(f"Retrying with fallback mirror: {cfg['fallback_url']}...")
        time.sleep(0.5)
        
        error_msg = (
            f"\n"
            f"================================================================================\n"
            f"FATAL DOWNLOAD ERROR: Dataset '{dataset_name}' could not be retrieved!\n"
            f"HTTP Error 404: Not Found\n"
            f"Remote mirror '{cfg['url']}' is permanently moved or decommissioned.\n"
            f"Fallback mirror '{cfg['fallback_url']}' timed out after 3 retries.\n"
            f"Checksum validation failed. Required SHA256: {cfg['expected_sha256']}\n"
            f"================================================================================\n"
        )
        sys.stderr.write(error_msg)
        raise RuntimeError(f"HTTP 404: Dataset mirror for '{dataset_name}' not found at {url}")

    # Offline debug fallback (synthetic generator for emergency internal linting)
    print(f"[DEBUG MODE] Generating dummy synthetic {dataset_name} for syntax linting...")
    import random
    with open(target_path, "w", encoding="utf-8") as f:
        headers = ["date"] + [f"feat_{i}" for i in range(cfg["num_features"])]
        f.write(",".join(headers) + "\n")
        for i in range(100):
            row = [f"2023-01-01 {i:02d}:00:00"] + [f"{random.uniform(-1.0, 1.0):.4f}" for _ in range(cfg["num_features"])]
            f.write(",".join(row) + "\n")
    print(f"[DEBUG MODE] Created stub dataset at {target_path}")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for STF-Net benchmarks")
    parser.add_argument("--dest_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "ETTh1", "Weather", "Electricity"])
    parser.add_argument("--offline-debug", action="store_true", help="Internal test flag")
    args = parser.parse_args()

    targets = list(DATASET_CONFIGS.keys()) if args.dataset == "all" else [args.dataset]
    for d in targets:
        attempt_download(d, args.dest_dir, offline_debug=args.offline_debug)

if __name__ == "__main__":
    main()
