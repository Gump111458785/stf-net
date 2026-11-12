import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import TimeSeriesDataset
from models.stf_net import STFNet
from models.baselines import DLinear, SimpleInformer
from utils.logger import setup_logger
from utils.metrics import compute_metrics

CHANNEL_MAP = {
    "ETTh1": 7,
    "Weather": 21,
    "Electricity": 321
}

def parse_args():
    parser = argparse.ArgumentParser(description="STF-Net Time Series Training Pipeline")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--dataset", type=str, default="ETTh1", choices=["ETTh1", "Weather", "Electricity"])
    parser.add_argument("--model", type=str, default="stf_net", choices=["stf_net", "dlinear", "informer"])
    parser.add_argument("--seq_len", type=int, default=96, help="Lookback window length")
    parser.add_argument("--pred_len", type=int, default=96, help="Forecasting horizon length")
    parser.add_argument("--modes", type=int, default=32, help="Fourier frequency modes")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3 for rapid evaluation <=15min)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (must be cuda)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    return parser.parse_args()

def check_hardware_requirements(device_arg):
    # Strict Hardware Requirement: CUDA GPU is required
    if not torch.cuda.is_available():
        raise RuntimeError(
            "\n"
            "======================================================================\n"
            "HARDWARE ERROR: STF-Net requires an NVIDIA CUDA-enabled GPU to run.\n"
            "torch.cuda.is_available() returned False.\n"
            "CPU execution is disabled for high-throughput Fourier tensor operations.\n"
            "======================================================================\n"
        )
    print(f"[Hardware Check Passed] Using CUDA Device: {torch.cuda.get_device_name(0)}")

def main():
    args = parse_args()
    logger = setup_logger("STF-Net-Train")
    logger.info("Initializing STF-Net Training Pipeline...")
    logger.info(f"Target dataset: {args.dataset}, Lookback: {args.seq_len}, Horizon: {args.pred_len}")

    # Enforce GPU requirement
    check_hardware_requirements(args.device)
    device = torch.device("cuda")

    num_channels = CHANNEL_MAP[args.dataset]
    data_file = os.path.join(args.data_dir, f"{args.dataset}.csv")

    if not os.path.exists(data_file):
        logger.error(f"Benchmark file '{data_file}' not found. Please run scripts/download_data.sh first.")
        raise FileNotFoundError(f"Missing {data_file}")

    train_set = TimeSeriesDataset(data_file, seq_len=args.seq_len, pred_len=args.pred_len, split="train")
    val_set = TimeSeriesDataset(data_file, seq_len=args.seq_len, pred_len=args.pred_len, split="val")
    test_set = TimeSeriesDataset(data_file, seq_len=args.seq_len, pred_len=args.pred_len, split="test")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    if args.model == "stf_net":
        model = STFNet(seq_len=args.seq_len, pred_len=args.pred_len, num_channels=num_channels, modes=args.modes).to(device)
    elif args.model == "dlinear":
        model = DLinear(seq_len=args.seq_len, pred_len=args.pred_len, num_channels=num_channels).to(device)
    else:
        model = SimpleInformer(seq_len=args.seq_len, pred_len=args.pred_len, num_channels=num_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.save_dir, f"{args.model}_{args.dataset}_pl{args.pred_len}_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_set)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * len(x)
        val_loss /= len(val_set)

        elapsed = time.time() - start_time
        logger.info(f"Epoch [{epoch:02d}/{args.epochs:02d}] Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Time: {elapsed:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)

    # Evaluation on Test Set
    model.load_state_dict(torch.load(best_ckpt_path))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.append(out.cpu())
            trues.append(y.cpu())
    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    test_metrics = compute_metrics(preds, trues)
    logger.info(f"Test Results [{args.dataset} - Horizon {args.pred_len}]: {test_metrics}")

if __name__ == "__main__":
    main()
