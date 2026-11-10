# SparseTSF: Summary, Deliverables and Reproduction Results

**Paper:** *SparseTSF: Modeling Long-term Time Series Forecasting with \*1k\* Parameters*,
Shengsheng Lin, Weiwei Lin, Wentai Wu, Haojun Chen, Junjie Yang. **ICML 2024 (Oral)**.
PDF: `paper/SparseTSF_ICML2024.pdf` (arXiv 2405.00946)

---

## 1. The paper as an analogy: forecasting customers at a coffee shop

You want to predict customers **every hour for the next 4 days** (96 values) from the
**last 30 days** of hourly records (720 values).

- **Usual approach (Transformers such as PatchTST or Informer):** hire a big team of analysts
  who compare every hour against every other hour. That means millions of parameters.
- **SparseTSF's approach:** a smart manager notices that *every day has the same shape*
  (busy at 8 am, quiet at 3 pm). The daily shape is already known, so the only question is
  **how each hour of the day changes from one day to the next.**

| Manager's step | Paper's step |
|---|---|
| "On average we get 50 customers an hour"; work only with the ups and downs | Subtract the mean of the input window (instance normalization) |
| Blend each hour slightly with its neighbours so one odd hour doesn't mislead | Sliding aggregation: 1D convolution (kernel 2⌊w/2⌋+1) plus a residual |
| Make 24 small lists: "all 8 am values", "all 9 am values", … (30 values each) | **Downsample by the period w = 24** → 24 sub-series of length n = 30 |
| Learn **one rule** for how a list moves from day to day, and use it for every hour | **One linear layer n → m (30 → 4), shared across all 24 phases** |
| Merge the 24 hourly forecasts back into time order and add the average back | Upsample (interleave) and add the mean back |

**Why it works:** the daily pattern comes for free from how the data is split, so the model
only learns the slow trend across days. That needs **145–925 parameters** instead of
millions, and still matches or beats Transformers on ETTh1, ETTh2, Electricity and Traffic.

**Limitation (for the Discussion section):** it needs a clear, known period. Data without a
regular cycle, such as stock prices, gains little from splitting by period.

---

## 2. What has been implemented (deliverables)

| Marking item | Deliverable | Status |
|---|---|---|
| Paper | `paper/SparseTSF_ICML2024.pdf`, the full 16-page paper | ✅ |
| Dataset from the paper | `data/ETTh1.csv`, 17,420 hourly rows, 7 channels (same as paper Table 1) | ✅ |
| Implementation from scratch | `sparsetsf/model.py` (SparseTSF and the Linear baseline), `sparsetsf/data.py` (ETTh1 protocol), `sparsetsf/train.py` (training and evaluation). Written from the paper's equations; the official repo was only read to confirm settings, never run | ✅ |
| Small numerical example by hand (2 pts) | `sparsetsf/hand_example.py`: L = 8, w = 2, H = 4, every step printed, then the **same weights are loaded into our PyTorch model and give the identical output** | ✅ |
| Coded example run on a benchmark from the paper, apple-to-apple (3 pts) | 5 seeds × 4 horizons on ETTh1, same protocol as the paper (below) | ✅ |
| Evaluation measures | MSE and MAE on standardized data, as in the paper | ✅ |
| Ablation | Linear baseline (paper's Table 5) vs. SparseTSF | ✅ |
| Raw results | `results/*_etth1.json` (per seed) and `results/*_log.txt` | ✅ |

### Apple-to-apple settings (identical to the paper, Sec. 4.1 and App. A.2)
look-back L = 720 · horizons H ∈ {96, 192, 336, 720} · period w = 24 · Adam · lr 0.02 ·
batch 256 · MSE loss · 30 epochs · lr unchanged for 3 epochs then ×0.8 per epoch · early
stopping (patience 5) on validation MSE · best checkpoint restored ·
ETTh1 split 12/4/4 months (6:2:2) · z-score fitted on the training split ·
**5 runs (seeds 2023–2027)**, reported as mean ± std, as in the paper.

---

## 3. Does our implementation match the paper's benchmark?

### 3a. Main result: paper Table 2 (ETTh1 MSE, mean ± std over 5 runs)

| Horizon | Paper (Table 2) | **Ours** | Difference |
|---|---|---|---|
| 96  | 0.359 ± 0.006 | **0.357 ± 0.003** | −0.002 |
| 192 | 0.397 ± 0.002 | **0.395 ± 0.001** | −0.002 |
| 336 | 0.404 ± 0.001 | **0.402 ± 0.002** | −0.002 |
| 720 | 0.417 ± 0.001 | **0.417 ± 0.002** | 0.000 |

**✅ It matches.** Every horizon is within 0.002 of the paper, inside or at the edge of the
paper's own run-to-run spread.
This table uses the paper's original evaluation, which drops the last incomplete test batch
(see 3b).

### 3b. Corrected evaluation (authors' later README results)
After publication, the authors fixed a "long-standing bug" in the shared LTSF code
framework: the test loader dropped the last incomplete batch (`drop_last=True`). Their README
reports new numbers with every test window evaluated. We compute both versions.

| Horizon | README MSE | **Our MSE** | README MAE | **Our MAE** |
|---|---|---|---|---|
| 96  | 0.362 | **0.366** | 0.388 | **0.393** |
| 192 | 0.403 | **0.401** | 0.411 | **0.411** |
| 336 | 0.434 | **0.430** | 0.428 | **0.427** |
| 720 | 0.426 | **0.426** | 0.447 | **0.447** |

**✅ It matches** within ±0.005. The README values were read from a GitHub mirror of the
official repository; confirm them against github.com/lss-1138/SparseTSF.

### 3c. Ablation: does the Sparse technique help? (paper Table 5, ETTh1 MSE)

| Horizon | Paper Linear | Our Linear | Paper SparseTSF | Our SparseTSF |
|---|---|---|---|---|
| 96  | 0.371 | 0.372 | 0.359 | 0.357 |
| 192 | 0.460 | 0.412 | 0.397 | 0.395 |
| 336 | 0.417 | 0.435 | 0.404 | 0.402 |
| 720 | 0.424 | 0.457 | 0.417 | 0.417 |

**⚠️ This only partly matches.** The paper's main conclusion holds: SparseTSF beats the plain
Linear model at every horizon. Our Linear numbers, however, only match the paper at H = 96.
The paper does not fully describe its Linear baseline (for example, whether it has a bias
term or uses the same lr), so we can't replicate it exactly. The paper's own Linear value at
H = 192 (0.460) is also higher than its H = 336 value (0.417), which is unusual. Treat this
table as a qualitative check, not an exact reproduction.

### Model size (paper Theorem 3.1: 2⌊w/2⌋+1 + ⌊L/w⌋·⌊H/w⌋ parameters)
Ours: **145 / 265 / 445 / 925** parameters for H = 96 / 192 / 336 / 720. The 925 for H = 720
matches the number the authors report.

### Runtime (CPU only, 16 cores, no GPU)
- SparseTSF: 20 runs in **4.4 min** total (about 13 s per run).
- Linear baseline: 20 runs in **2.0 min** total (about 6 s per run).
- Full reproduction of both models: **about 6–7 minutes**.

---

## 4. Small numerical example (`sparsetsf/hand_example.py`)
Input `x = [1, 3, 2, 4, 3, 5, 4, 6]` (odd hours rise 1→4, even hours rise 3→6), w = 2,
forecast 4 steps; the true continuation is `[5, 7, 6, 8]`.

1. Mean = 3.5 → x − mean = [−2.5, −0.5, −1.5, 0.5, −0.5, 1.5, 0.5, 2.5]
2. Conv with k = [0.25, 0.5, 0.25] plus the residual → [−3.875, −1.75, −2.25, 0.25, −0.25, 2.25, 1.75, 3.875]
3. Downsample to 2 sub-series (one per phase) of length 4
4. Shared linear W = [[0, 0, −1, 2], [0, 0, −2, 3]] ("continue the trend") → 2 values per phase
5. Interleave the phases and add the mean back → forecast [7.25, 9, 9.25, 10.625]; MSE 6.63, MAE 2.53
6. Our PyTorch `SparseTSF` with the same weights gives **the identical output** ✅
7. With the conv weights set to zero, the 4 shared weights alone continue **both** phases
   exactly → [5, 7, 6, 8], MSE = 0. This is the paper's core idea in miniature.

---

## 5. How to reproduce
```bash
cd /home/ubuntu/DL-Paper
.venv/bin/python sparsetsf/hand_example.py                 # numerical example
.venv/bin/python sparsetsf/train.py --model sparsetsf      # main result (~4.5 min on CPU)
.venv/bin/python sparsetsf/train.py --model linear         # ablation baseline (~2 min)
```
