/* Tiny dependency-free SVG charts: line chart (crosshair tooltip), heatmap (cell tooltip),
 * signed bar chart (bar tooltip) and dot plot (dot tooltip). Colours come from CSS
 * custom properties, so light/dark themes are handled by style.css.
 */
(function (global) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function niceTicks(min, max, count) {
    const span = max - min || 1, raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => span / s <= count) || mag * 10;
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function fmt(v, d = 3) { return (Math.abs(v) < 1e-12 ? 0 : v).toFixed(d); }

  /* ---------- tooltip shared by all charts in a container ---------- */
  function tooltip(container) {
    container.style.position = "relative";
    let t = container.querySelector(".tip");
    if (!t) { t = document.createElement("div"); t.className = "tip"; container.appendChild(t); }
    return {
      show(html, x, y) {
        t.innerHTML = html; t.style.display = "block";
        const cw = container.clientWidth, tw = t.offsetWidth;
        t.style.left = Math.min(Math.max(x + 14, 0), cw - tw - 4) + "px";
        t.style.top = Math.max(y - t.offsetHeight - 10, 0) + "px";
      },
      hide() { t.style.display = "none"; },
    };
  }

  function frame(container, height, margin) {
    container.querySelectorAll("svg").forEach(s => s.remove());
    const width = Math.max(container.clientWidth, 280);
    const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img" });
    container.prepend(svg);
    return { svg, width, height, iw: width - margin.l - margin.r, ih: height - margin.t - margin.b };
  }

  /* ---------- line chart ---------- */
  function line(container, opt) {
    const m = { t: 14, r: 16, b: 34, l: 46 };
    const { svg, iw, ih } = frame(container, opt.height || 300, m);
    const g = el("g", { transform: `translate(${m.l},${m.t})` }, svg);
    const all = opt.series.flatMap(s => s.points);
    const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    const pad = (y1 - y0) * 0.08 || 1; y0 -= pad; y1 += pad;
    const sx = v => ((v - x0) / (x1 - x0 || 1)) * iw;
    const sy = v => ih - ((v - y0) / (y1 - y0)) * ih;

    niceTicks(y0, y1, 5).forEach(v => {
      el("line", { x1: 0, x2: iw, y1: sy(v), y2: sy(v), class: "grid" }, g);
      el("text", { x: -8, y: sy(v), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, g)
        .textContent = opt.yFmt ? opt.yFmt(v) : v;
    });
    niceTicks(x0, x1, Math.max(3, Math.floor(iw / 90))).forEach(v => {
      el("text", { x: sx(v), y: ih + 18, class: "tick", "text-anchor": "middle" }, g)
        .textContent = opt.xFmt ? opt.xFmt(v) : v;
    });
    el("line", { x1: 0, x2: iw, y1: ih, y2: ih, class: "axis" }, g);
    if (opt.xLabel) el("text", { x: iw, y: ih + 32, class: "axis-label", "text-anchor": "end" }, g).textContent = opt.xLabel;

    if (opt.band) {
      el("rect", { x: sx(opt.band[0]), y: 0, width: sx(opt.band[1]) - sx(opt.band[0]), height: ih, class: "band" }, g);
    }
    if (opt.vline != null) {
      el("line", { x1: sx(opt.vline.x), x2: sx(opt.vline.x), y1: 0, y2: ih, class: "vline" }, g);
      el("text", { x: sx(opt.vline.x) + 6, y: 10, class: "vline-label" }, g).textContent = opt.vline.label;
    }

    opt.series.forEach(s => {
      const d = s.points.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1)).join("");
      el("path", { d, class: "series", stroke: css(s.color), "stroke-dasharray": s.dash || "" }, g);
      if (s.markers) s.points.forEach(p =>
        el("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 4, fill: css(s.color), class: "marker" }, g));
    });

    // crosshair + tooltip
    const tip = tooltip(container);
    const cross = el("line", { y1: 0, y2: ih, class: "crosshair", visibility: "hidden" }, g);
    const dots = opt.series.map(s => el("circle", { r: 4, fill: css(s.color), class: "hover-dot", visibility: "hidden" }, g));
    const hit = el("rect", { width: iw, height: ih, fill: "transparent" }, g);
    hit.addEventListener("pointermove", ev => {
      const r = hit.getBoundingClientRect();
      const xv = x0 + ((ev.clientX - r.left) / iw) * (x1 - x0);
      let rows = [];
      opt.series.forEach((s, i) => {
        let best = null;
        for (const p of s.points) if (!best || Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
        const near = best && Math.abs(best[0] - xv) <= (x1 - x0) / Math.max(s.points.length, 40) * 2;
        if (near) {
          dots[i].setAttribute("cx", sx(best[0])); dots[i].setAttribute("cy", sy(best[1]));
          dots[i].setAttribute("visibility", "visible");
          rows.push(`<span class="sw" style="background:${css(s.color)}"></span>${s.name}: <b>${fmt(best[1])}</b>`);
        } else dots[i].setAttribute("visibility", "hidden");
      });
      const xr = Math.round(xv);
      cross.setAttribute("x1", sx(xr)); cross.setAttribute("x2", sx(xr)); cross.setAttribute("visibility", "visible");
      const cr = container.getBoundingClientRect();
      tip.show(`<div class="tip-h">${opt.xFmt ? opt.xFmt(xr, true) : xr}</div>${rows.join("<br>")}`,
        ev.clientX - cr.left, ev.clientY - cr.top);
    });
    hit.addEventListener("pointerleave", () => {
      tip.hide(); cross.setAttribute("visibility", "hidden");
      dots.forEach(d => d.setAttribute("visibility", "hidden"));
    });
  }

  /* ---------- diverging colour (blue <- gray -> red) ---------- */
  function hexToRgb(h) { h = h.replace("#", ""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
  function mix(a, b, t) { return a.map((v, i) => Math.round(v + (b[i] - v) * t)); }
  function diverging(v, maxAbs) {
    const t = Math.max(-1, Math.min(1, v / (maxAbs || 1)));
    const mid = hexToRgb(css("--div-mid")), neg = hexToRgb(css("--div-neg")), pos = hexToRgb(css("--div-pos"));
    const c = t < 0 ? mix(mid, neg, -t) : mix(mid, pos, t);
    return `rgb(${c.join(",")})`;
  }

  /* ---------- heatmap ---------- */
  function heatmap(container, opt) {
    const rows = opt.matrix.length, cols = opt.matrix[0].length;
    const m = { t: 8, r: 8, b: opt.xTitle ? 34 : 18, l: opt.yTitle ? 48 : 36 };
    const width = Math.max(container.clientWidth, 260);
    const cell = Math.min((width - m.l - m.r) / cols, opt.maxCell || 22);
    const ch = opt.cellHeight || Math.min(cell, 14);
    const { svg } = frame(container, m.t + m.b + rows * ch, m);
    const g = el("g", { transform: `translate(${m.l},${m.t})` }, svg);
    const maxAbs = opt.maxAbs || Math.max(...opt.matrix.flat().map(Math.abs));
    const tip = tooltip(container);

    opt.matrix.forEach((row, r) => row.forEach((v, c) => {
      const rect = el("rect", { x: c * cell, y: r * ch, width: Math.max(cell - 1, 1), height: Math.max(ch - 1, 1),
        rx: cell > 8 ? 2 : 0, fill: diverging(v, maxAbs), class: "cell" }, g);
      rect.addEventListener("pointermove", ev => {
        const cr = container.getBoundingClientRect();
        tip.show(opt.tip(r, c, v), ev.clientX - cr.left, ev.clientY - cr.top);
      });
      rect.addEventListener("pointerleave", () => tip.hide());
    }));
    const every = opt.rowEvery || 1;
    for (let r = 0; r < rows; r += every)
      el("text", { x: -6, y: r * ch + ch / 2, class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, g)
        .textContent = opt.rowLabel ? opt.rowLabel(r) : r;
    const ce = opt.colEvery || Math.max(1, Math.ceil(cols / 10));
    for (let c = 0; c < cols; c += ce)
      el("text", { x: c * cell + cell / 2, y: rows * ch + 12, class: "tick", "text-anchor": "middle" }, g)
        .textContent = opt.colLabel ? opt.colLabel(c) : c;
    if (opt.xTitle) el("text", { x: 0, y: rows * ch + 28, class: "axis-label" }, g).textContent = opt.xTitle;
    if (opt.yTitle) el("text", { x: -m.l + 10, y: (rows * ch) / 2, class: "axis-label",
      transform: `rotate(-90 ${-m.l + 10} ${(rows * ch) / 2})`, "text-anchor": "middle" }, g).textContent = opt.yTitle;
  }

  /* ---------- signed bar chart (e.g. conv kernel weights) ---------- */
  function bars(container, opt) {
    const m = { t: 10, r: 10, b: 30, l: 46 };
    const { svg, iw, ih } = frame(container, opt.height || 170, m);
    const g = el("g", { transform: `translate(${m.l},${m.t})` }, svg);
    const v = opt.values, lo = Math.min(0, ...v), hi = Math.max(0, ...v);
    const pad = (hi - lo) * 0.1 || 1;
    const sy = x => ih - ((x - (lo - pad)) / (hi - lo + 2 * pad)) * ih;
    const bw = iw / v.length;
    niceTicks(lo - pad, hi + pad, 4).forEach(t => {
      el("line", { x1: 0, x2: iw, y1: sy(t), y2: sy(t), class: "grid" }, g);
      el("text", { x: -8, y: sy(t), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, g).textContent = t;
    });
    el("line", { x1: 0, x2: iw, y1: sy(0), y2: sy(0), class: "axis" }, g);
    const tip = tooltip(container);
    v.forEach((x, i) => {
      const y = Math.min(sy(x), sy(0)), h = Math.max(Math.abs(sy(x) - sy(0)), 1);
      el("rect", { x: i * bw + 1, y, width: Math.max(bw - 2, 1), height: h, rx: Math.min(4, bw / 3),
        fill: css(opt.color || "--series-1"), class: "bar" }, g);
      const hit = el("rect", { x: i * bw, y: 0, width: bw, height: ih, fill: "transparent" }, g);
      hit.addEventListener("pointermove", ev => {
        const cr = container.getBoundingClientRect();
        tip.show(opt.tip(i, x), ev.clientX - cr.left, ev.clientY - cr.top);
      });
      hit.addEventListener("pointerleave", () => tip.hide());
      if (i % Math.max(1, Math.ceil(v.length / 12)) === 0)
        el("text", { x: i * bw + bw / 2, y: ih + 16, class: "tick", "text-anchor": "middle" }, g)
          .textContent = opt.label ? opt.label(i) : i;
    });
  }

  /* ---------- dot plot: categories x two series, optional error bars ---------- */
  function dots(container, opt) {
    const m = { t: 14, r: 16, b: 34, l: 50 };
    const { svg, iw, ih } = frame(container, opt.height || 260, m);
    const g = el("g", { transform: `translate(${m.l},${m.t})` }, svg);
    const vals = opt.series.flatMap(s => s.values.map((v, i) => [v - (s.err ? s.err[i] : 0), v + (s.err ? s.err[i] : 0)])).flat();
    let y0 = Math.min(...vals), y1 = Math.max(...vals); const pad = (y1 - y0) * 0.15 || 0.01; y0 -= pad; y1 += pad;
    const sy = v => ih - ((v - y0) / (y1 - y0)) * ih;
    const bw = iw / opt.categories.length;
    niceTicks(y0, y1, 5).forEach(t => {
      el("line", { x1: 0, x2: iw, y1: sy(t), y2: sy(t), class: "grid" }, g);
      el("text", { x: -8, y: sy(t), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, g).textContent = t.toFixed(2);
    });
    el("line", { x1: 0, x2: iw, y1: ih, y2: ih, class: "axis" }, g);
    const tip = tooltip(container);
    opt.categories.forEach((c, i) => {
      el("text", { x: bw * i + bw / 2, y: ih + 18, class: "tick", "text-anchor": "middle" }, g).textContent = c;
      const n = opt.series.length;
      opt.series.forEach((s, j) => {
        const cx = bw * i + bw / 2 + (j - (n - 1) / 2) * 18, v = s.values[i];
        if (s.err) el("line", { x1: cx, x2: cx, y1: sy(v - s.err[i]), y2: sy(v + s.err[i]), stroke: css(s.color), class: "err" }, g);
        el("circle", { cx, cy: sy(v), r: 5, fill: css(s.color), class: "marker" }, g);
      });
      const hit = el("rect", { x: bw * i, y: 0, width: bw, height: ih, fill: "transparent" }, g);
      hit.addEventListener("pointermove", ev => {
        const cr = container.getBoundingClientRect();
        tip.show(`<div class="tip-h">${opt.xTitle || ""} ${c}</div>` + opt.series.map(s =>
          `<span class="sw" style="background:${css(s.color)}"></span>${s.name}: <b>${s.values[i].toFixed(3)}</b>` +
          (s.err ? ` ± ${s.err[i].toFixed(3)}` : "")).join("<br>"), ev.clientX - cr.left, ev.clientY - cr.top);
      });
      hit.addEventListener("pointerleave", () => tip.hide());
    });
    if (opt.xTitle) el("text", { x: iw, y: ih + 32, class: "axis-label", "text-anchor": "end" }, g).textContent = opt.xTitle;
  }

  global.Charts = { line, heatmap, bars, dots, fmt };
})(window);
