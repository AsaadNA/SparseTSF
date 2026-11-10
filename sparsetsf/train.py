"""Train and evaluate SparseTSF (and the Linear baseline) on ETTh1 under the paper's protocol.

Paper settings (Sec. 4.1, App. A.2 and the official ETTh1 script):
  look-back 720, horizons {96, 192, 336, 720}, period w = 24, Adam, lr 0.02, batch 256,
  MSE loss, 30 epochs, lr kept for 3 epochs then x0.8 per epoch, early stopping
  (patience 5) on validation MSE, best checkpoint restored, 5 runs (seeds 2023-2027).

Two test numbers are reported:
  * paper   - last incomplete test batch dropped (drop_last=True). This is how the numbers in
              the paper's Table 2 were produced (the "long-standing bug" the authors fixed later).
  * full    - every test window evaluated (drop_last=False). This is the corrected protocol;
              its values match the authors' later README table.

Usage:  python sparsetsf/train.py [--model sparsetsf|linear] [--horizons 96 192] [--seeds 5]
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ETTh1Windows
from model import LinearBaseline, SparseTSF, count_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_at(epoch, base_lr):
    """Learning rate used after finishing `epoch` (1-indexed); same rule as the paper."""
    return base_lr if epoch < 3 else base_lr * 0.8 ** (epoch - 3)


@torch.no_grad()
def mean_batch_mse(model, loader):
    model.eval()
    losses = [torch.mean((model(x) - y) ** 2).item() for x, y in loader]
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def test_metrics(model, dataset, batch_size, drop_last):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)
    preds, trues = [], []
    for x, y in loader:
        preds.append(model(x).numpy())
        trues.append(y.numpy())
    p, t = np.concatenate(preds), np.concatenate(trues)
    return float(np.mean((p - t) ** 2)), float(np.mean(np.abs(p - t)))


def run(model_name, horizon, seed, args, return_model=False):
    set_seed(seed)
    csv = os.path.join(ROOT, "data", "ETTh1.csv")
    train_set = ETTh1Windows(csv, "train", args.seq_len, horizon)
    val_set = ETTh1Windows(csv, "val", args.seq_len, horizon)
    test_set = ETTh1Windows(csv, "test", args.seq_len, horizon)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = (SparseTSF(args.seq_len, horizon, args.period) if model_name == "sparsetsf"
             else LinearBaseline(args.seq_len, horizon))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_state, bad_epochs, epochs_run = float("inf"), None, 0, 0
    for epoch in range(1, args.epochs + 1):
        epochs_run = epoch
        for x, y in train_loader:
            opt.zero_grad()
            loss = torch.mean((model(x) - y) ** 2)
            loss.backward()
            opt.step()

        val = mean_batch_mse(model, val_loader)
        if val < best_val:
            best_val, best_state, bad_epochs = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
        for g in opt.param_groups:
            g["lr"] = lr_at(epoch, args.lr)

    model.load_state_dict(best_state)
    mse_p, mae_p = test_metrics(model, test_set, args.batch_size, drop_last=True)
    mse_f, mae_f = test_metrics(model, test_set, args.batch_size, drop_last=False)
    result = dict(model=model_name, horizon=horizon, seed=seed, params=count_params(model),
                  epochs=epochs_run, val_mse=best_val,
                  mse_paper=mse_p, mae_paper=mae_p, mse_full=mse_f, mae_full=mae_f)
    return (result, model) if return_model else result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sparsetsf", choices=["sparsetsf", "linear"])
    ap.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seq_len", type=int, default=720)
    ap.add_argument("--period", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    results = []
    for h in args.horizons:
        for seed in range(2023, 2023 + args.seeds):
            t0 = time.time()
            r = run(args.model, h, seed, args)
            results.append(r)
            print(f"{r['model']:9s} H={h:3d} seed={seed} params={r['params']:6d} "
                  f"epochs={r['epochs']:2d}  MSE(paper)={r['mse_paper']:.4f} "
                  f"MSE(full)={r['mse_full']:.4f} MAE(full)={r['mae_full']:.4f} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", f"{args.model}_etth1.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)

    print(f"\nSummary over {args.seeds} seeds (mean ± std)")
    for h in args.horizons:
        rs = [r for r in results if r["horizon"] == h]
        f = lambda k: f"{np.mean([r[k] for r in rs]):.3f}±{np.std([r[k] for r in rs]):.3f}"
        print(f"H={h:3d}  MSE(paper) {f('mse_paper')}  MAE(paper) {f('mae_paper')}  "
              f"MSE(full) {f('mse_full')}  MAE(full) {f('mae_full')}")
    print("saved", out)


if __name__ == "__main__":
    main()
