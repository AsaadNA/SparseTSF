# SparseTSF Interactive Demo

A static web page that runs our from-scratch SparseTSF (trained on ETTh1) **inside the browser**
and visualises every step of the model.

## Open it
Double-click `index.html`, or open it in any modern browser. No server, internet connection or
installation is needed: all data is in `data/model_data.js`.

If you prefer serving it:
```bash
cd /home/ubuntu/DL-Paper/webapp && python3 -m http.server 8000   # then open http://localhost:8000
```

## Sections
| Section | What you can do |
|---|---|
| **Idea** | The coffee-shop analogy, step by step |
| **Live forecast** | Choose a horizon (96–720 h), a channel and a test window (slider, Random, ▶ Play); see the forecast vs. ground truth and MSE/MAE |
| **Inside the model** | For the same window: normalisation → learned conv kernel → 24×30 downsampled matrix → shared weights W → output matrix (hover over everything) |
| **Hand example** | The report's numerical example; edit x, w, m, k and W and every step updates (presets included) |
| **Parameters** | Calculator for paper Theorem 3.1 vs. a plain Linear model |
| **Results** | Paper vs. our reproduction (5 seeds), plus the Linear ablation |

The page checks itself on load: it reruns test window 0 in JavaScript and compares the result with the
PyTorch output saved at export time. The "✓ Verified" line under the live chart shows this check.

## Structure
```
webapp/
├── index.html               page layout
├── assets/css/style.css     styles (light + dark theme tokens)
├── assets/js/sparsetsf.js   SparseTSF forward pass in JS (mirrors sparsetsf/model.py)
├── assets/js/charts.js      dependency-free SVG charts with tooltips
├── assets/js/app.js         UI logic
├── data/model_data.js       generated: trained weights, ETTh1 test data, results
└── scripts/export_data.py   regenerates data/model_data.js
```

## Regenerate the data
Re-trains SparseTSF (seed 2023, paper settings) for each horizon on CPU in about 1 minute, and picks up
the latest `results/*.json`:
```bash
cd /home/ubuntu/DL-Paper
.venv/bin/python webapp/scripts/export_data.py
```
Values in the demo are z-score standardised with training-split statistics, the same space the paper
reports MSE/MAE in.
