"""ETTh1 loading with the standard LTSF protocol used by the paper (FITS/Autoformer split):
12 months train / 4 months val / 4 months test (6:2:2 of the first 20 months),
z-score standardization fitted on the training split only, sliding windows of stride 1.
Metrics are reported in the standardized space, as in the paper.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

MONTH = 30 * 24  # hours


class ETTh1Windows(Dataset):
    def __init__(self, csv_path: str, split: str, seq_len: int, pred_len: int):
        df = pd.read_csv(csv_path)
        values = df.iloc[:, 1:].values.astype(np.float64)   # 7 channels, drop 'date'

        starts = [0, 12 * MONTH - seq_len, 16 * MONTH - seq_len]
        ends = [12 * MONTH, 16 * MONTH, 20 * MONTH]
        i = {"train": 0, "val": 1, "test": 2}[split]

        train = values[:12 * MONTH]
        mu, sd = train.mean(axis=0), train.std(axis=0)       # population std (= sklearn)
        data = (values - mu) / sd

        self.x = torch.tensor(data[starts[i]:ends[i]], dtype=torch.float32)
        self.L, self.H = seq_len, pred_len

    def __len__(self):
        return len(self.x) - self.L - self.H + 1

    def __getitem__(self, idx):
        return self.x[idx: idx + self.L], self.x[idx + self.L: idx + self.L + self.H]
