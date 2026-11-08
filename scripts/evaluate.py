import os
import argparse
import json
import torch
from torch.utils.data import DataLoader

from data.dataset import TimeSeriesDataset
from models.stf_net import STFNet
from models.baselines import DLinear, SimpleInformer
from utils.logger import setup_logger
from utils.metrics import compute_metrics

CHANNEL_MAP = {"ETTh1": 7, "Weather": 21, "Electricity": 321}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate STF-Net models")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--model", type=str, default="stf_net")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_json", type=str, default="./results/eval_results.json")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("STF-Net-Eval")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU device is required for STF-Net evaluation.")

    device = torch.device("cuda")
    num_channels = CHANNEL_MAP[args.dataset]
    data_file = os.path.join(args.data_dir, f"{args.dataset}.csv")

    test_set = TimeSeriesDataset(data_file, seq_len=args.seq_len, pred_len=args.pred_len, split="test")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    if args.model == "stf_net":
        model = STFNet(seq_len=args.seq_len, pred_len=args.pred_len, num_channels=num_channels).to(device)
    elif args.model == "dlinear":
        model = DLinear(seq_len=args.seq_len, pred_len=args.pred_len, num_channels=num_channels).to(device)
    else:
        model = SimpleInformer(seq_len=args.seq_len, pred_len=args.pred_len, num_channels=num_channels).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds.append(model(x).cpu())
            trues.append(y.cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    metrics = compute_metrics(preds, trues)
    logger.info(f"Evaluation Metrics [{args.dataset} - Horizon {args.pred_len}]: {metrics}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
