import os
import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    """
    Sliding window dataset for multivariate time series forecasting.
    """
    def __init__(self, data_path, seq_len=96, pred_len=96, split="train", scale=True):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split
        self.scale = scale

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"[STF-Net Dataset Error] Raw benchmark file '{data_path}' was not found!\n"
                f"Please ensure you run `bash scripts/download_data.sh` to fetch the benchmark datasets."
            )

        # Parse CSV file
        raw_data = []
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(",")
                # Skip date column at index 0, parse float values
                vals = [float(x) for x in parts[1:]]
                raw_data.append(vals)

        data = np.array(raw_data, dtype=np.float32)
        n = len(data)

        # 70% train, 10% val, 20% test
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)

        train_data = data[:train_end]
        self.mean = train_data.mean(axis=0)
        self.std = train_data.std(axis=0) + 1e-5

        if self.scale:
            data = (data - self.mean) / self.std

        if split == "train":
            self.data = data[:train_end]
        elif split == "val":
            self.data = data[train_end - seq_len : val_end]
        else:
            self.data = data[val_end - seq_len :]

        self.num_samples = len(self.data) - self.seq_len - self.pred_len + 1
        if self.num_samples < 0:
            self.num_samples = 0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)
