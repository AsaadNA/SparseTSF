/* SparseTSF forward pass for ONE channel, in plain JavaScript.
 * Mirrors sparsetsf/model.py exactly (PyTorch Conv1d = cross-correlation, zero padding).
 * Returns every intermediate so the page can visualise each step.
 */
(function (global) {
  "use strict";

  function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }

  /**
   * @param {number[]} x       look-back window, length L
   * @param {number[]} kernel  conv weights, length 2*floor(w/2)+1
   * @param {number[][]} W     shared linear weights, shape m x n
   * @param {number} w         period
   */
  function forward(x, kernel, W, w) {
    const L = x.length, n = L / w, m = W.length, H = m * w, p = Math.floor(w / 2);
    if (!Number.isInteger(n)) throw new Error(`look-back length ${L} is not a multiple of w=${w}`);
    if (kernel.length !== 2 * p + 1) throw new Error(`kernel must have ${2 * p + 1} weights`);
    if (W.some(r => r.length !== n)) throw new Error(`each weight row must have ${n} values`);

    // 1) instance normalisation
    const mu = mean(x);
    const xn = x.map(v => v - mu);

    // 2) sliding aggregation (zero padding) + residual
    const conv = xn.map((_, t) => {
      let s = 0;
      for (let j = 0; j < kernel.length; j++) {
        const idx = t + j - p;
        if (idx >= 0 && idx < L) s += kernel[j] * xn[idx];
      }
      return s;
    });
    const agg = xn.map((v, t) => v + conv[t]);

    // 3) downsample: sub[phase][segment] = agg[segment * w + phase]   (w x n)
    const sub = Array.from({ length: w }, (_, ph) =>
      Array.from({ length: n }, (_, seg) => agg[seg * w + ph]));

    // 4) shared linear n -> m on every phase   (w x m)
    const ys = sub.map(row => W.map(wr => wr.reduce((s, wv, i) => s + wv * row[i], 0)));

    // 5) upsample (interleave phases back into time order) + 6) de-normalise
    const yNorm = new Array(H);
    for (let j = 0; j < m; j++) for (let ph = 0; ph < w; ph++) yNorm[j * w + ph] = ys[ph][j];
    const y = yNorm.map(v => v + mu);

    return { L, n, m, H, w, mu, xn, conv, agg, sub, ys, yNorm, y };
  }

  function mse(a, b) { return a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length; }
  function mae(a, b) { return a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0) / a.length; }

  /** Parameters of SparseTSF (paper Theorem 3.1) and of a plain Linear L->H model. */
  function paramCount(L, H, w) {
    const conv = 2 * Math.floor(w / 2) + 1;
    const linear = Math.floor(L / w) * Math.floor(H / w);
    return { conv, linear, total: conv + linear, plainLinear: L * H + H };
  }

  global.SparseTSF = { forward, mse, mae, paramCount, mean };
})(typeof window !== "undefined" ? window : globalThis);
