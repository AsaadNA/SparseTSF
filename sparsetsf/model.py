"""SparseTSF (Lin et al., ICML 2024), written from scratch from the paper (Sec. 3).

Shapes:  B = batch, L = look-back, H = horizon, C = channels, w = period,
         n = L // w (segments in input), m = H // w (segments in output).
"""
import torch.nn as nn


class SparseTSF(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, period: int):
        super().__init__()
        assert seq_len % period == 0 and pred_len % period == 0
        self.L, self.H, self.w = seq_len, pred_len, period
        self.n, self.m = seq_len // period, pred_len // period

        # Sliding aggregation (Eq. 2): kernel 2*floor(w/2)+1, zero padding, no bias.
        self.agg = nn.Conv1d(1, 1, kernel_size=2 * (period // 2) + 1,
                             padding=period // 2, bias=False)
        # Sparse sliding prediction (Eq. 3): ONE linear map n -> m shared by all w phases.
        self.linear = nn.Linear(self.n, self.m, bias=False)

    def forward(self, x):                                   # x: (B, L, C)
        B, L, C = x.shape
        # 1) instance normalization: remove each window's mean (per channel)
        mean = x.mean(dim=1, keepdim=True)                  # (B, 1, C)
        x = (x - mean).permute(0, 2, 1).reshape(B * C, 1, L)  # channel independence

        # 2) sliding aggregation with a residual connection
        x = self.agg(x) + x                                 # (B*C, 1, L)

        # 3) downsample: split the series into n periods of length w,
        #    then group by phase -> w sub-series, each of length n
        x = x.reshape(B * C, self.n, self.w).permute(0, 2, 1)   # (B*C, w, n)

        # 4) cross-period forecast with the shared linear layer
        y = self.linear(x)                                  # (B*C, w, m)

        # 5) upsample: interleave the phases back into time order
        y = y.permute(0, 2, 1).reshape(B, C, self.H)        # (B, C, H)

        # 6) undo normalization
        return y.permute(0, 2, 1) + mean                    # (B, H, C)


class LinearBaseline(nn.Module):
    """Ablation baseline from the paper's Table 5: a plain Linear model (L -> H) with the
    same channel independence and instance normalization, but without the Sparse technique."""

    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):                                   # (B, L, C)
        mean = x.mean(dim=1, keepdim=True)
        y = self.linear((x - mean).permute(0, 2, 1))        # (B, C, H)
        return y.permute(0, 2, 1) + mean


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
