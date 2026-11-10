/* UI wiring for the SparseTSF demo. Depends on: SPARSETSF_DATA, SparseTSF, Charts. */
(function () {
  "use strict";
  const D = window.SPARSETSF_DATA, S = window.SparseTSF, C = window.Charts;
  const L = D.meta.seq_len, W = D.meta.period, NCH = D.series.length;
  const T0 = new Date(D.meta.test_start.replace(" ", "T") + "Z");
  const $ = id => document.getElementById(id);
  const state = { h: 96, ch: NCH - 1, idx: 0, full: false, timer: null };

  /* ---------- helpers ---------- */
  const pad2 = v => String(v).padStart(2, "0");
  function dateAt(offsetHours) {
    const d = new Date(T0.getTime() + offsetHours * 3600e3);
    return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())} ${pad2(d.getUTCHours())}:00`;
  }
  const hourOfDay = offsetHours => (T0.getUTCHours() + offsetHours) % 24;
  const nWindows = h => D.series[0].length - L - h + 1;
  const model = () => D.models[String(state.h)];
  const stat = (v, l) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  const vec = (a, d = 3) => "[" + a.map(v => C.fmt(v, d)).join(", ") + "]";

  function windowData(ch) {
    const s = D.series[ch];
    return { x: s.slice(state.idx, state.idx + L), y: s.slice(state.idx + L, state.idx + L + state.h) };
  }

  /* ---------- hero ---------- */
  function hero() {
    const r = D.results.sparsetsf, p = D.results.paper;
    const params = Object.values(D.models).map(m => m.params);
    $("hero-stats").innerHTML =
      stat(`${Math.min(...params)}–${Math.max(...params)}`, "learnable parameters (H = 96 → 720)") +
      (r ? stat(`${r["96"].mse_paper[0].toFixed(3)}`, `our MSE @96 · paper ${p.sparsetsf_mse["96"].toFixed(3)}`) : "") +
      stat("in-browser", "every forecast is computed live in JavaScript") +
      stat("CPU", "trained without a GPU in about 13 s per run");
  }

  /* ---------- live explorer ---------- */
  function setupExplorer() {
    D.meta.channels.forEach((c, i) => $("channel").add(new Option(c + (c === "OT" ? " (oil temperature)" : ""), i)));
    $("channel").value = state.ch;
    $("horizon").onchange = e => { state.h = +e.target.value; clampIdx(); renderAll(); };
    $("channel").onchange = e => { state.ch = +e.target.value; renderAll(); };
    $("window").oninput = e => { state.idx = +e.target.value; renderAll(); };
    $("fullback").onchange = e => { state.full = e.target.checked; renderExplorer(); };
    $("random").onclick = () => { state.idx = Math.floor(Math.random() * nWindows(state.h)); renderAll(); };
    $("play").onclick = togglePlay;
  }
  function clampIdx() {
    const n = nWindows(state.h);
    $("window").max = n - 1;
    state.idx = Math.min(state.idx, n - 1);
  }
  function togglePlay() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; $("play").textContent = "▶ Play"; return; }
    $("play").textContent = "❚❚ Pause";
    state.timer = setInterval(() => {
      state.idx += 24;
      if (state.idx >= nWindows(state.h)) state.idx = 0;
      renderAll();
    }, 160);
  }

  function renderExplorer() {
    const M = model(), { x, y } = windowData(state.ch);
    const r = S.forward(x, M.conv, M.linear, W);
    const back = state.full ? L : 168;
    const actual = [];
    for (let t = -back; t < 0; t++) actual.push([t, x[L + t]]);
    y.forEach((v, t) => actual.push([t, v]));
    const fc = r.y.map((v, t) => [t, v]);
    const base = state.idx + L;
    C.line($("chart-main"), {
      height: 320, series: [{ name: "Actual", color: "--series-1", points: actual },
                            { name: "Forecast", color: "--series-2", points: fc }],
      vline: { x: 0, label: "forecast starts" }, band: [0, state.h - 1],
      xLabel: "hours relative to forecast start (standardised values)",
      xFmt: (v, long) => long ? `${v >= 0 ? "+" : ""}${v} h · ${dateAt(base + v)}` : (v > 0 ? "+" : "") + v,
    });

    // metrics: this channel and all channels for this window
    let mseAll = 0, maeAll = 0;
    for (let c = 0; c < NCH; c++) {
      const w = windowData(c), f = S.forward(w.x, M.conv, M.linear, W).y;
      mseAll += S.mse(f, w.y) / NCH; maeAll += S.mae(f, w.y) / NCH;
    }
    $("win-label").textContent = `#${state.idx} / ${nWindows(state.h) - 1} · starts ${dateAt(base)}`;
    $("window").value = state.idx;
    $("metrics").innerHTML =
      `<span>This channel: MSE <b>${S.mse(r.y, y).toFixed(3)}</b> · MAE <b>${S.mae(r.y, y).toFixed(3)}</b></span>` +
      `<span>All 7 channels, this window: MSE <b>${mseAll.toFixed(3)}</b> · MAE <b>${maeAll.toFixed(3)}</b></span>` +
      `<span>Whole test set (seed 2023): MSE <b>${M.test_mse.toFixed(3)}</b> · MAE <b>${M.test_mae.toFixed(3)}</b></span>`;
    return r;
  }

  function verify() {
    const out = Object.entries(D.models).map(([h, M]) => {
      let worst = 0;
      for (let c = 0; c < NCH; c++) {
        const y = S.forward(D.series[c].slice(0, L), M.conv, M.linear, W).y;
        y.forEach((v, i) => { worst = Math.max(worst, Math.abs(v - M.ref_window0[c][i])); });
      }
      return worst;
    });
    const worst = Math.max(...out);
    $("verify").innerHTML = worst < 1e-3
      ? `<span class="ok">✓ Verified:</span> the JavaScript forward pass matches the PyTorch model on test window 0 for every horizon and channel (max difference ${worst.toExponential(1)}, from rounding the exported values).`
      : `<span class="err-msg">✗ JavaScript and PyTorch outputs differ by ${worst.toFixed(4)}.</span>`;
  }

  /* ---------- pipeline ---------- */
  function renderPipeline(r) {
    const M = model(), base = state.idx;
    $("s1-mean").textContent = r.mu.toFixed(3);
    C.line($("chart-s1"), {
      height: 180, series: [{ name: "x′ = x − mean", color: "--series-1", points: r.xn.map((v, t) => [t - L, v]) }],
      xLabel: "hours before forecast start", xFmt: (v, long) => long ? `${v} h · ${dateAt(base + L + v)}` : v,
    });
    C.bars($("chart-kernel"), {
      values: M.conv, label: i => i - Math.floor(W / 2), height: 180,
      tip: (i, v) => `offset ${i - Math.floor(W / 2)} h: <b>${v.toFixed(4)}</b>`,
    });
    const last = 168;
    C.line($("chart-s2"), {
      height: 180, xLabel: "last week (hours)",
      series: [{ name: "x′", color: "--series-3", points: r.xn.slice(-last).map((v, t) => [t - last, v]) },
               { name: "x″", color: "--series-1", points: r.agg.slice(-last).map((v, t) => [t - last, v]) }],
      xFmt: (v, long) => long ? `${v} h · ${dateAt(base + L + v)}` : v,
    });

    const hourLabel = ph => pad2(hourOfDay(base + ph)) + ":00";
    C.heatmap($("chart-s3"), {
      matrix: r.sub, maxCell: 30, cellHeight: 13, rowEvery: 3,
      rowLabel: hourLabel, colLabel: c => c - r.n, colEvery: 5,
      xTitle: "day (relative to forecast start)", yTitle: "hour",
      tip: (ph, seg, v) => `${hourLabel(ph)} on day ${seg - r.n}<br>${dateAt(base + seg * W + ph)}<br>x″ = <b>${v.toFixed(3)}</b>`,
    });

    $("s4-shape").textContent = `${r.m} × ${r.n}`;
    $("s4-m").textContent = r.m;
    C.heatmap($("chart-W"), {
      matrix: M.linear, maxCell: 22, cellHeight: Math.max(8, Math.min(18, 220 / r.m)),
      rowLabel: j => "+" + (j + 1), rowEvery: Math.max(1, Math.ceil(r.m / 8)),
      colLabel: i => i - r.n, colEvery: 5, xTitle: "input day", yTitle: "out day",
      tip: (j, i, v) => `output day +${j + 1} ← input day ${i - r.n}<br>weight <b>${v.toFixed(4)}</b>`,
    });
    C.heatmap($("chart-s4"), {
      matrix: r.ys, maxCell: 30, cellHeight: 13, rowEvery: 3,
      rowLabel: hourLabel, colLabel: j => "+" + (j + 1), colEvery: Math.max(1, Math.ceil(r.m / 8)),
      xTitle: "future day", yTitle: "hour",
      tip: (ph, j, v) => `${hourLabel(ph)} on day +${j + 1}<br>${dateAt(base + L + j * W + ph)}<br>forecast (normalised) <b>${v.toFixed(3)}</b>`,
    });
    $("s5-params").textContent = `${M.params} (conv ${M.conv.length} + linear ${r.m}×${r.n} = ${r.m * r.n})`;
  }

  /* ---------- playground ---------- */
  const PRESETS = {
    report: { x: "1, 3, 2, 4, 3, 5, 4, 6", y: "5, 7, 6, 8", w: 2, m: 2, k: [0.25, 0.5, 0.25], W: [[0, 0, -1, 2], [0, 0, -2, 3]] },
    nok:    { x: "1, 3, 2, 4, 3, 5, 4, 6", y: "5, 7, 6, 8", w: 2, m: 2, k: [0, 0, 0], W: [[0, 0, -1, 2], [0, 0, -2, 3]] },
    avg:    { x: "1, 3, 2, 4, 3, 5, 4, 6", y: "5, 7, 6, 8", w: 2, m: 2, k: [0, 0, 0], W: [[0, 0, 0.5, 0.5], [0, 0, 0.5, 0.5]] },
  };
  const parseList = s => s.split(/[\s,;]+/).filter(Boolean).map(Number);

  function buildGrid(id, rows, cols, values) {
    const box = $(id);
    box.style.gridTemplateColumns = `repeat(${cols}, max-content)`;
    box.innerHTML = "";
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const inp = document.createElement("input");
      inp.type = "number"; inp.step = "0.05";
      inp.value = values && values[r] && values[r][c] != null ? values[r][c] : 0;
      inp.dataset.r = r; inp.dataset.c = c; inp.oninput = renderPlayground;
      inp.setAttribute("aria-label", `${id} row ${r + 1} column ${c + 1}`);
      box.appendChild(inp);
    }
  }
  function readGrid(id, rows, cols) {
    const out = Array.from({ length: rows }, () => new Array(cols).fill(0));
    $(id).querySelectorAll("input").forEach(i => { out[+i.dataset.r][+i.dataset.c] = +i.value || 0; });
    return out;
  }
  function pgDims() {
    const x = parseList($("pg-x").value), w = +$("pg-w").value, m = +$("pg-m").value;
    return { x, w, m, n: x.length / w, klen: 2 * Math.floor(w / 2) + 1 };
  }
  function rebuildPlayground(preset) {
    const d = pgDims();
    if (!Number.isInteger(d.n)) return renderPlayground();
    const oldK = preset ? [preset.k] : [readGrid("pg-k", 1, $("pg-k").children.length)[0]];
    const oldW = preset ? preset.W : null;
    buildGrid("pg-k", 1, d.klen, oldK);
    buildGrid("pg-W", d.m, d.n, oldW);
    renderPlayground();
  }
  function applyPreset(name) {
    const p = PRESETS[name];
    $("pg-x").value = p.x; $("pg-y").value = p.y; $("pg-w").value = p.w; $("pg-m").value = p.m;
    rebuildPlayground(p);
  }
  function matrixHtml(rows) { return rows.map(r => vec(r)).join("<br>"); }

  function renderPlayground() {
    const d = pgDims(), out = $("pg-steps");
    if (!d.x.length || d.x.some(isNaN)) { out.innerHTML = `<p class="err-msg">Enter numbers separated by commas.</p>`; return; }
    if (!Number.isInteger(d.n)) { out.innerHTML = `<p class="err-msg">The length of x (${d.x.length}) must be a multiple of w = ${d.w}.</p>`; return; }
    if ($("pg-W").children.length !== d.m * d.n || $("pg-k").children.length !== d.klen) return rebuildPlayground();
    const k = readGrid("pg-k", 1, d.klen)[0], Wm = readGrid("pg-W", d.m, d.n);
    const r = S.forward(d.x, k, Wm, d.w);
    const truth = parseList($("pg-y").value);
    const hasTruth = truth.length === r.H && !truth.some(isNaN);
    const row = (key, val) => `<div class="row"><div class="k">${key}</div><div class="val">${val}</div></div>`;
    out.innerHTML =
      row(`dimensions`, `L = ${r.L}, w = ${d.w}, n = L/w = ${r.n}, m = ${r.m}, H = m·w = ${r.H}, params = ${k.length + r.m * r.n}`) +
      row(`1 · mean = ${C.fmt(r.mu)} → x − mean`, vec(r.xn)) +
      row(`2 · conv1d(x′, k) (zero padding ${Math.floor(d.w / 2)})`, vec(r.conv)) +
      row(`&nbsp;&nbsp;&nbsp;x″ = conv + x′`, vec(r.agg)) +
      row(`3 · downsample → ${d.w} sub-series × ${r.n}`, matrixHtml(r.sub)) +
      row(`4 · shared linear: each row × Wᵀ → ${d.w} × ${r.m}`, matrixHtml(r.ys)) +
      row(`5 · interleave → normalised forecast`, vec(r.yNorm)) +
      row(`6 · + mean → forecast ŷ`, `<b>${vec(r.y)}</b>`) +
      (hasTruth ? row(`7 · vs truth ${vec(truth, 2)}`, `MSE = <b>${S.mse(r.y, truth).toFixed(4)}</b> · MAE = <b>${S.mae(r.y, truth).toFixed(4)}</b>`)
                : row(`7 · metrics`, `enter ${r.H} ground-truth values to compute MSE / MAE`));
    const actual = d.x.map((v, t) => [t, v]).concat(hasTruth ? truth.map((v, t) => [r.L + t, v]) : []);
    C.line($("chart-pg"), {
      height: 200, xLabel: "time step",
      series: [{ name: "Input / truth", color: "--series-1", points: actual, markers: true },
               { name: "Forecast", color: "--series-2", points: r.y.map((v, t) => [r.L + t, v]), markers: true }],
      vline: { x: r.L - 0.5, label: "forecast" },
      xFmt: v => `t = ${v}`,
    });
  }
  function setupPlayground() {
    document.querySelectorAll("[data-preset]").forEach(b => b.onclick = () => applyPreset(b.dataset.preset));
    $("pg-x").oninput = () => rebuildPlayground();
    $("pg-w").onchange = () => rebuildPlayground();
    $("pg-m").onchange = () => rebuildPlayground();
    $("pg-y").oninput = renderPlayground;
    applyPreset("report");
  }

  /* ---------- parameter calculator ---------- */
  function renderParams() {
    const Lv = +$("pc-L").value, Hv = +$("pc-H").value, w = +$("pc-w").value;
    $("pc-Lv").textContent = Lv; $("pc-Hv").textContent = Hv;
    const p = S.paramCount(Lv, Hv, w);
    const warn = Math.floor(Hv / w) === 0 ? " (H is shorter than one period!)" : "";
    $("pc-out").innerHTML =
      stat(p.total.toLocaleString(), `SparseTSF parameters (conv ${p.conv} + linear ${Math.floor(Lv / w)}×${Math.floor(Hv / w)})${warn}`) +
      stat(p.plainLinear.toLocaleString(), "plain Linear L→H parameters") +
      stat(p.total ? Math.round(p.plainLinear / p.total).toLocaleString() + "×" : "–", "fewer parameters") +
      stat("6.31 M", "PatchTST, for comparison (paper Table 3)");
  }
  function setupParams() { ["pc-L", "pc-H", "pc-w"].forEach(id => $(id).oninput = renderParams); renderParams(); }

  /* ---------- results ---------- */
  function renderResults() {
    const ours = D.results.sparsetsf, lin = D.results.linear, P = D.results.paper;
    const hs = ["96", "192", "336", "720"];
    if (!ours) { $("res-table").innerHTML = "<tr><td>Run sparsetsf/train.py, then re-run the export script.</td></tr>"; return; }
    C.dots($("chart-res"), {
      categories: hs, xTitle: "horizon",
      series: [{ name: "Paper", color: "--series-1", values: hs.map(h => P.sparsetsf_mse[h]), err: hs.map(h => P.sparsetsf_std[h]) },
               { name: "Ours", color: "--series-2", values: hs.map(h => ours[h].mse_paper[0]), err: hs.map(h => ours[h].mse_paper[1]) }],
    });
    const pm = v => `${v[0].toFixed(3)} ± ${v[1].toFixed(3)}`;
    $("res-table").innerHTML =
      `<tr><th>Horizon</th><th>Paper MSE</th><th>Ours MSE (paper protocol)</th><th>Δ</th><th>Ours MSE (all windows)</th><th>Ours MAE (all windows)</th></tr>` +
      hs.map(h => {
        let dlt = ours[h].mse_paper[0] - P.sparsetsf_mse[h];
        if (Math.abs(dlt) < 0.0005) dlt = 0;
        return `<tr><td>${h}</td><td>${P.sparsetsf_mse[h].toFixed(3)} ± ${P.sparsetsf_std[h].toFixed(3)}</td><td><b>${pm(ours[h].mse_paper)}</b></td>` +
          `<td class="${Math.abs(dlt) <= 0.005 ? "ok" : ""}">${dlt >= 0 ? "+" : ""}${dlt.toFixed(3)}</td>` +
          `<td>${pm(ours[h].mse_full)}</td><td>${pm(ours[h].mae_full)}</td></tr>`;
      }).join("");
    $("abl-table").innerHTML =
      `<tr><th>Horizon</th><th>Paper Linear</th><th>Our Linear</th><th>Paper SparseTSF</th><th>Our SparseTSF</th><th>Sparse helps (ours)?</th></tr>` +
      hs.map(h => {
        const ol = lin ? lin[h].mse_paper[0] : NaN, os = ours[h].mse_paper[0];
        return `<tr><td>${h}</td><td>${P.linear_mse[h].toFixed(3)}</td><td>${isNaN(ol) ? "–" : ol.toFixed(3)}</td>` +
          `<td>${P.sparsetsf_mse[h].toFixed(3)}</td><td>${os.toFixed(3)}</td>` +
          `<td class="${os < ol ? "ok" : ""}">${isNaN(ol) ? "–" : (os < ol ? "✓ " : "✗ ") + ((1 - os / ol) * 100).toFixed(1) + "% lower MSE"}</td></tr>`;
      }).join("");
  }

  /* ---------- theme ---------- */
  function setupTheme() {
    try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage blocked */ }
    $("theme").onclick = () => {
      const dark = document.documentElement.dataset.theme
        ? document.documentElement.dataset.theme === "dark"
        : matchMedia("(prefers-color-scheme: dark)").matches;
      document.documentElement.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch (e) { /* ignore */ }
      renderEverything();
    };
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderEverything);
  }

  /* ---------- render ---------- */
  function renderAll() { const r = renderExplorer(); renderPipeline(r); }
  function renderEverything() { renderAll(); renderPlayground(); renderResults(); }

  setupTheme();
  hero();
  setupExplorer();
  clampIdx();
  setupPlayground();
  setupParams();
  renderAll();
  renderResults();
  verify();
  let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(renderEverything, 150); });
})();
