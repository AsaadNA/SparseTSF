"""SparseTSF - small numerical example that can be done by hand (one channel).

  look-back L = 8, period w = 2, horizon H = 4  ->  n = L/w = 4,  m = H/w = 2
  conv kernel size = 2*floor(w/2)+1 = 3, padding 1

Step 1 computes everything with plain numbers (NumPy), the way you would on paper.
Step 2 loads the SAME weights into our PyTorch SparseTSF model and checks that it gives
the identical forecast, which shows that the from-scratch implementation does what the
hand calculation does.
"""
import numpy as np
import torch

from model import SparseTSF

np.set_printoptions(precision=4, suppress=True)

x = np.array([1., 3., 2., 4., 3., 5., 4., 6.])      # odd hours: 1,2,3,4  even hours: 3,4,5,6
y_true = np.array([5., 7., 6., 8.])                  # the pattern continues
k = np.array([0.25, 0.5, 0.25])                      # conv kernel (a smoothing filter)
W = np.array([[0., 0., -1., 2.],                     # next period  = 2*last - previous
              [0., 0., -2., 3.]])                    # period after = 3*last - 2*previous

print("input x            :", x)

# 1) instance normalization
mu = x.mean()
xn = x - mu
print(f"1) mean = {mu}   x - mean :", xn)

# 2) sliding aggregation (zero padding) + residual
pad = np.concatenate([[0.], xn, [0.]])
conv = np.array([pad[t:t + 3] @ k for t in range(len(xn))])
xa = conv + xn
print("2) conv(x)          :", conv)
print("   conv(x) + x      :", xa)

# 3) downsample: reshape to n x w, transpose to w x n (one row per phase)
sub = xa.reshape(4, 2).T
print("3) sub-series (w x n):\n", sub)

# 4) shared linear layer n -> m applied to every phase
ys = sub @ W.T
print("4) per-phase forecast (w x m):\n", ys)

# 5) upsample (interleave phases) + 6) add the mean back
y_pred = ys.T.reshape(-1) + mu
print("5-6) forecast       :", y_pred, "  truth:", y_true)

mse = np.mean((y_pred - y_true) ** 2)
mae = np.mean(np.abs(y_pred - y_true))
print(f"7) MSE = {mse:.4f}   MAE = {mae:.4f}")

# ---- same weights in our PyTorch model --------------------------------------
model = SparseTSF(seq_len=8, pred_len=4, period=2)
with torch.no_grad():
    model.agg.weight.copy_(torch.tensor(k).view(1, 1, 3))
    model.linear.weight.copy_(torch.tensor(W))
    out = model(torch.tensor(x, dtype=torch.float32).view(1, 8, 1)).view(-1).numpy()
print("\nPyTorch SparseTSF output:", out,
      " matches hand calculation:", np.allclose(out, y_pred, atol=1e-5))

# ---- insight: the weights above are hand-picked, not trained -----------------
# The smoothing filter plus the residual roughly doubles the deviations, so the forecast
# overshoots. With the conv switched off (k = 0), the 4 shared linear weights alone continue
# BOTH phases' trends exactly, which is the core idea of the paper.
with torch.no_grad():
    model.agg.weight.zero_()
    out0 = model(torch.tensor(x, dtype=torch.float32).view(1, 8, 1)).view(-1).numpy()
print("with k = 0           :", out0, f"  MSE = {np.mean((out0 - y_true) ** 2):.4f}")
