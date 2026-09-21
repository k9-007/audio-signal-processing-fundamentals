/* Interactive widgets for the audio-fundamentals notes.
 * Vanilla JS, no dependencies. Each widget is a function (container) => void,
 * registered in WIDGETS and mounted on <div class="widget" data-widget="name">.
 * Inlined into the HTML by scripts/build_html_notes.py.
 */
(function () {
  "use strict";

  /* ───────────────────────── core helpers ───────────────────────── */

  const PALETTE = {
    bg: "#0b0e14", grid: "#1f2533", axis: "#4b5468", text: "#c8cedb",
    blue: "#7aa2f7", green: "#9ece6a", orange: "#ff9e64", red: "#f7768e",
    purple: "#bb9af7", yellow: "#e0af68", cyan: "#7dcfff",
  };

  function h(tag, attrs = {}, children = []) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") e.className = v;
      else if (k === "html") e.innerHTML = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    for (const c of [].concat(children)) {
      if (c == null) continue;
      e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return e;
  }

  function slider(parent, { label, min, max, step, value, fmt = (v) => v }, onChange) {
    const out = h("span", { class: "w-val" }, fmt(value));
    const input = h("input", { type: "range", min, max, step, value });
    const row = h("label", { class: "w-slider" }, [h("span", { class: "w-label" }, label), input, out]);
    parent.appendChild(row);
    input.addEventListener("input", () => {
      out.textContent = fmt(parseFloat(input.value));
      onChange(parseFloat(input.value));
    });
    return { get: () => parseFloat(input.value), set: (v) => { input.value = v; out.textContent = fmt(v); } };
  }

  function toggle(parent, label, checked, onChange) {
    const input = h("input", { type: "checkbox" });
    input.checked = checked;
    input.addEventListener("change", () => onChange(input.checked));
    parent.appendChild(h("label", { class: "w-toggle" }, [input, h("span", {}, label)]));
    return () => input.checked;
  }

  function button(parent, label, onClick, cls = "") {
    const b = h("button", { class: "w-btn " + cls, onclick: onClick }, label);
    parent.appendChild(b);
    return b;
  }

  function makeCanvas(parent, height) {
    const cv = h("canvas", { class: "w-canvas" });
    parent.appendChild(cv);
    const ctx = cv.getContext("2d");
    function size() {
      const dpr = window.devicePixelRatio || 1;
      const w = cv.clientWidth || parent.clientWidth || 600;
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(height * dpr);
      cv.style.height = height + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { W: w, H: height };
    }
    return { cv, ctx, size };
  }

  function clear(ctx, W, H) {
    ctx.fillStyle = PALETTE.bg;
    ctx.fillRect(0, 0, W, H);
  }

  /* Plot a polyline given data arrays and axis ranges into rect {x,y,w,h}. */
  function plotLine(ctx, xs, ys, rect, range, color, width = 1.6, dash = []) {
    const { x0, x1, y0, y1 } = range;
    ctx.save();
    ctx.beginPath();
    ctx.rect(rect.x, rect.y, rect.w, rect.h);
    ctx.clip();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
      const px = rect.x + ((xs[i] - x0) / (x1 - x0)) * rect.w;
      const py = rect.y + rect.h - ((ys[i] - y0) / (y1 - y0)) * rect.h;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.restore();
  }

  function axes(ctx, rect, range, { xlabel = "", ylabel = "", xticks = 5, yticks = 4, xfmt = (v) => v, yfmt = (v) => v } = {}) {
    const { x0, x1, y0, y1 } = range;
    ctx.save();
    ctx.strokeStyle = PALETTE.grid;
    ctx.lineWidth = 1;
    ctx.fillStyle = PALETTE.text;
    ctx.font = "11px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    for (let i = 0; i <= xticks; i++) {
      const v = x0 + (i / xticks) * (x1 - x0);
      const px = rect.x + (i / xticks) * rect.w;
      ctx.beginPath(); ctx.moveTo(px, rect.y); ctx.lineTo(px, rect.y + rect.h); ctx.stroke();
      ctx.fillText(xfmt(v), px, rect.y + rect.h + 14);
    }
    ctx.textAlign = "right";
    for (let i = 0; i <= yticks; i++) {
      const v = y0 + (i / yticks) * (y1 - y0);
      const py = rect.y + rect.h - (i / yticks) * rect.h;
      ctx.beginPath(); ctx.moveTo(rect.x, py); ctx.lineTo(rect.x + rect.w, py); ctx.stroke();
      ctx.fillText(yfmt(v), rect.x - 6, py + 4);
    }
    ctx.strokeStyle = PALETTE.axis;
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
    ctx.textAlign = "center";
    ctx.fillStyle = PALETTE.text;
    if (xlabel) ctx.fillText(xlabel, rect.x + rect.w / 2, rect.y + rect.h + 28);
    if (ylabel) {
      ctx.save(); ctx.translate(rect.x - 44, rect.y + rect.h / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillText(ylabel, 0, 0); ctx.restore();
    }
    ctx.restore();
  }

  function title(ctx, text, x, y, color = PALETTE.text) {
    ctx.save(); ctx.fillStyle = color; ctx.font = "bold 12px -apple-system, sans-serif";
    ctx.textAlign = "left"; ctx.fillText(text, x, y); ctx.restore();
  }

  /* In-place iterative radix-2 FFT. re/im are Float64Array of power-of-2 length. */
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let j = 0; j < len / 2; j++) {
          const ur = re[i + j], ui = im[i + j];
          const vr = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci;
          const vi = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr;
          re[i + j] = ur + vr; im[i + j] = ui + vi;
          re[i + j + len / 2] = ur - vr; im[i + j + len / 2] = ui - vi;
          const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
        }
      }
    }
  }

  function magnitudeSpectrum(signal) {
    let n = 1; while (n < signal.length) n <<= 1;
    const re = new Float64Array(n), im = new Float64Array(n);
    re.set(signal);
    fft(re, im);
    const half = n / 2 + 1, mag = new Float64Array(half);
    for (let k = 0; k < half; k++) mag[k] = Math.hypot(re[k], im[k]) / (signal.length / 2);
    return mag;
  }

  /* Inferno-like colormap, v in [0,1]. */
  function cmap(v) {
    v = Math.min(1, Math.max(0, v));
    const stops = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
    const p = v * (stops.length - 1), i = Math.floor(p), f = p - i;
    const a = stops[i], b = stops[Math.min(i + 1, stops.length - 1)];
    return `rgb(${a[0] + (b[0] - a[0]) * f | 0},${a[1] + (b[1] - a[1]) * f | 0},${a[2] + (b[2] - a[2]) * f | 0})`;
  }

  function heatmap(ctx, matrix, rect, vmin, vmax) {
    const rows = matrix.length, cols = matrix[0].length;
    const cw = rect.w / cols, ch = rect.h / rows;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        ctx.fillStyle = cmap((matrix[r][c] - vmin) / (vmax - vmin));
        ctx.fillRect(rect.x + c * cw, rect.y + rect.h - (r + 1) * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }
    ctx.strokeStyle = PALETTE.axis;
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
  }

  function readout(parent) {
    const r = h("div", { class: "w-readout" });
    parent.appendChild(r);
    return (html) => { r.innerHTML = html; };
  }

  function frame(container, titleText, help) {
    container.innerHTML = "";
    container.appendChild(h("div", { class: "w-title" }, [h("span", { class: "w-badge" }, "interactive"), titleText]));
    if (help) container.appendChild(h("p", { class: "w-help" }, help));
    const controls = h("div", { class: "w-controls" });
    const body = h("div", { class: "w-body" });
    container.appendChild(controls);
    container.appendChild(body);
    return { controls, body };
  }

  const fmt = {
    hz: (v) => `${(+v).toFixed(v < 10 ? 1 : 0)} Hz`,
    int: (v) => `${Math.round(v)}`,
    f2: (v) => (+v).toFixed(2),
    pct: (v) => `${Math.round(v * 100)}%`,
  };

  const WIDGETS = {};

  /* ───────────────────────── 1. sine explorer ───────────────────────── */
  WIDGETS["sine-explorer"] = function (container) {
    const { controls, body } = frame(container, "Sine wave explorer — amplitude, frequency, phase",
      "Drag the sliders and watch how each symbol in y(t) = A·sin(2πft + φ) changes the wave. Count the cycles in one second.");
    const { ctx, size } = makeCanvas(body, 240);
    const out = readout(body);
    let A = 0.8, f = 3, phi = 0;
    const draw = () => {
      const { W, H } = size(); clear(ctx, W, H);
      const rect = { x: 50, y: 20, w: W - 70, h: H - 60 };
      const range = { x0: 0, x1: 1, y0: -1.1, y1: 1.1 };
      axes(ctx, rect, range, { xlabel: "time (seconds)", ylabel: "amplitude", xfmt: (v) => v.toFixed(2), yfmt: (v) => v.toFixed(1) });
      const N = 800, xs = [], ys = [];
      for (let i = 0; i <= N; i++) { const t = i / N; xs.push(t); ys.push(A * Math.sin(2 * Math.PI * f * t + phi)); }
      plotLine(ctx, xs, ys, rect, range, PALETTE.blue, 2);
      // mark peaks to make cycle counting easy
      ctx.fillStyle = PALETTE.orange;
      for (let k = 0; k < f + 1; k++) {
        const tPeak = ((Math.PI / 2 - phi) / (2 * Math.PI) + k) / f;
        if (tPeak >= 0 && tPeak <= 1) {
          const px = rect.x + tPeak * rect.w, py = rect.y + rect.h - ((A + 1.1) / 2.2) * rect.h;
          ctx.beginPath(); ctx.arc(px, py, 4, 0, 2 * Math.PI); ctx.fill();
        }
      }
      out(`<b>y(t) = ${A.toFixed(2)} · sin(2π · ${f.toFixed(1)} · t + ${phi.toFixed(2)})</b><br>
           Cycles in one second: <b>${f.toFixed(1)}</b> (orange dots = peaks) &nbsp;·&nbsp; period T = 1/f = <b>${(1 / f).toFixed(3)} s</b>
           &nbsp;·&nbsp; ${f < 20 ? "below 20 Hz you would feel this as vibration, not hear it" : "audible"}`);
    };
    slider(controls, { label: "A (amplitude)", min: 0, max: 1, step: 0.01, value: A, fmt: fmt.f2 }, (v) => { A = v; draw(); });
    slider(controls, { label: "f (frequency)", min: 0.5, max: 12, step: 0.5, value: f, fmt: fmt.hz }, (v) => { f = v; draw(); });
    slider(controls, { label: "φ (phase)", min: 0, max: 6.28, step: 0.01, value: phi, fmt: (v) => `${v.toFixed(2)} rad` }, (v) => { phi = v; draw(); });
    draw(); window.addEventListener("resize", draw);
  };

  /* ───────────────────────── 2. sampling & aliasing ───────────────────────── */
  WIDGETS["sampling-aliasing"] = function (container) {
    const { controls, body } = frame(container, "Sampling & aliasing — the spinning-fan illusion",
      "The blue wave is the real sound. Dots are what the microphone records at the chosen sampling rate. The dashed orange wave is the frequency a computer would *infer* from the dots alone. Push the sampling rate below 2·f and watch the alias appear.");
    const { ctx, size } = makeCanvas(body, 250);
    const out = readout(body);
    let f = 8, fs = 30;
    const draw = () => {
      const { W, H } = size(); clear(ctx, W, H);
      const rect = { x: 50, y: 20, w: W - 70, h: H - 60 };
      const range = { x0: 0, x1: 1, y0: -1.2, y1: 1.2 };
      axes(ctx, rect, range, { xlabel: "time (seconds)", xfmt: (v) => v.toFixed(2), yfmt: (v) => v.toFixed(1) });
      const N = 1200, xs = [], ys = [];
      for (let i = 0; i <= N; i++) { const t = i / N; xs.push(t); ys.push(Math.sin(2 * Math.PI * f * t)); }
      plotLine(ctx, xs, ys, rect, range, PALETTE.blue, 1.5);
      const nyq = fs / 2;
      // alias frequency: fold f into [0, fs/2]
      let fa = Math.abs(f - Math.round(f / fs) * fs);
      const aliased = f > nyq;
      if (aliased) {
        const ya = xs.map((t) => Math.sin(2 * Math.PI * fa * t) * (Math.sin(2 * Math.PI * f * (0)) >= 0 ? 1 : 1));
        // choose sign so alias passes through the samples: sin(2π f t_k) vs sin(2π fa t_k)
        const t0 = 1 / fs; const s = Math.sign(Math.sin(2 * Math.PI * f * t0) * Math.sin(2 * Math.PI * fa * t0)) || 1;
        plotLine(ctx, xs, ya.map((v) => s * v), rect, range, PALETTE.orange, 2, [6, 4]);
      }
      ctx.fillStyle = PALETTE.yellow;
      for (let k = 0; k * (1 / fs) <= 1; k++) {
        const t = k / fs, y = Math.sin(2 * Math.PI * f * t);
        const px = rect.x + t * rect.w, py = rect.y + rect.h - ((y + 1.2) / 2.4) * rect.h;
        ctx.beginPath(); ctx.arc(px, py, 4, 0, 2 * Math.PI); ctx.fill();
      }
      out(`Sampling rate fs = <b>${fs}</b> samples/s → Nyquist limit fs/2 = <b>${nyq}</b> Hz &nbsp;·&nbsp; true frequency f = <b>${f} Hz</b> &nbsp;·&nbsp; samples per cycle = <b>${(fs / f).toFixed(2)}</b><br>
           ${aliased
          ? `<span class="bad">ALIASED</span> — f &gt; fs/2. The dots are indistinguishable from a <b>${fa.toFixed(1)} Hz</b> wave (orange). The recording now *contains* a frequency that never existed.`
          : `<span class="good">SAFE</span> — f &lt; fs/2, at least 2 samples per cycle. The dots uniquely determine the blue wave.`}`);
    };
    slider(controls, { label: "true frequency f", min: 1, max: 30, step: 1, value: f, fmt: fmt.hz }, (v) => { f = v; draw(); });
    slider(controls, { label: "sampling rate fs", min: 4, max: 80, step: 1, value: fs, fmt: (v) => `${v} samples/s` }, (v) => { fs = v; draw(); });
    draw(); window.addEventListener("resize", draw);
  };

  /* ───────────────────────── 3. Fourier mixer ───────────────────────── */
  WIDGETS["fourier-mixer"] = function (container) {
    const { controls, body } = frame(container, "Fourier mixer — un-mixing the paint, live",
      "Mix up to three sinusoids. The left plot is what you would record (the messy sum); the right plot is the FFT magnitude — it un-mixes the recipe back into spikes. Set an amplitude to 0 to remove a component.");
    const { ctx, size } = makeCanvas(body, 260);
    const out = readout(body);
    const comps = [{ f: 5, a: 1 }, { f: 20, a: 0.6 }, { f: 40, a: 0.3 }];
    const FS = 256, N = 512; // 2 s at 256 samples/s → 0.5 Hz bins, Nyquist 128
    const draw = () => {
      const { W, H } = size(); clear(ctx, W, H);
      const half = (W - 90) / 2;
      const r1 = { x: 45, y: 20, w: half, h: H - 65 }, r2 = { x: 45 + half + 40, y: 20, w: half, h: H - 65 };
      const sig = new Float64Array(N), xs = [];
      for (let n = 0; n < N; n++) { const t = n / FS; xs.push(t); let v = 0; for (const c of comps) v += c.a * Math.sin(2 * Math.PI * c.f * t); sig[n] = v; }
      const amax = Math.max(1, comps.reduce((s, c) => s + c.a, 0));
      const rng1 = { x0: 0, x1: 1, y0: -amax, y1: amax };
      axes(ctx, r1, rng1, { xlabel: "time (s) — what you record", xfmt: (v) => v.toFixed(2), yfmt: (v) => v.toFixed(1) });
      plotLine(ctx, xs, Array.from(sig), r1, rng1, PALETTE.blue, 1.4);
      const mag = magnitudeSpectrum(sig);
      const freqs = Array.from(mag, (_, k) => k * FS / N);
      const rng2 = { x0: 0, x1: 64, y0: 0, y1: Math.max(0.1, Math.max(...mag) * 1.1) };
      axes(ctx, r2, rng2, { xlabel: "frequency (Hz) — FFT magnitude", xticks: 8, xfmt: (v) => v.toFixed(0), yfmt: (v) => v.toFixed(1) });
      // draw spectrum as bars
      ctx.fillStyle = PALETTE.green;
      for (let k = 0; k < mag.length && freqs[k] <= 64; k++) {
        const px = r2.x + (freqs[k] / 64) * r2.w, bh = (mag[k] / rng2.y1) * r2.h;
        ctx.fillRect(px - 1, r2.y + r2.h - bh, 2.5, bh);
      }
      title(ctx, "time domain", r1.x, 14); title(ctx, "frequency domain", r2.x, 14);
      const active = comps.filter((c) => c.a > 0).map((c) => `${c.f} Hz (A=${c.a.toFixed(2)})`).join(" + ");
      out(`Recipe: <b>${active || "silence"}</b> → the FFT shows exactly one spike per ingredient, at the right frequency, with height ∝ amplitude.<br>
           <span class="muted">Sampled at ${FS} samples/s for 2 s → ${N} samples → frequency resolution ${(FS / N).toFixed(2)} Hz per bin, Nyquist ${FS / 2} Hz. Note: the FFT cannot tell you <i>when</i> each tone played — only <i>that</i> it played.</span>`);
    };
    comps.forEach((c, i) => {
      const row = h("div", { class: "w-group" }, h("span", { class: "w-group-title" }, `sine ${i + 1}`));
      controls.appendChild(row);
      slider(row, { label: "f", min: 1, max: 60, step: 1, value: c.f, fmt: fmt.hz }, (v) => { c.f = v; draw(); });
      slider(row, { label: "A", min: 0, max: 1, step: 0.05, value: c.a, fmt: fmt.f2 }, (v) => { c.a = v; draw(); });
    });
    draw(); window.addEventListener("resize", draw);
  };

  /* ───────────────────────── 4. STFT / spectrogram explorer ───────────────────────── */
  WIDGETS["stft-explorer"] = function (container) {
    const { controls, body } = frame(container, "STFT explorer — the time–frequency trade-off",
      "A synthetic 1-second clip: a rising chirp, a steady tone, and two sharp clicks. Change the window size and watch detail move between the axes: long windows sharpen frequency (thin lines) but smear time (fat clicks); short windows do the reverse.");
    const { ctx, size } = makeCanvas(body, 300);
    const out = readout(body);
    const FS = 8000, LEN = 8192;
    const sig = new Float64Array(LEN);
    for (let n = 0; n < LEN; n++) {
      const t = n / FS;
      sig[n] = 0.6 * Math.sin(2 * Math.PI * (300 + 1500 * t) * t) + 0.4 * Math.sin(2 * Math.PI * 1200 * t);
    }
    for (const c of [0.30, 0.31, 0.72]) { const n0 = Math.floor(c * FS); for (let k = 0; k < 8; k++) sig[n0 + k] += 1.2 * (k % 2 ? -1 : 1); }
    let win = 256;
    const compute = () => {
      const hop = win / 4, frames = Math.floor((LEN - win) / hop), bins = win / 2 + 1;
      const S = Array.from({ length: bins }, () => new Float64Array(frames));
      const re = new Float64Array(win), im = new Float64Array(win);
      const hann = Float64Array.from({ length: win }, (_, n) => 0.5 - 0.5 * Math.cos(2 * Math.PI * n / win));
      let vmax = -1e9;
      for (let t = 0; t < frames; t++) {
        for (let n = 0; n < win; n++) { re[n] = sig[t * hop + n] * hann[n]; im[n] = 0; }
        fft(re, im);
        for (let k = 0; k < bins; k++) { const db = 20 * Math.log10(Math.hypot(re[k], im[k]) / win + 1e-6); S[k][t] = db; if (db > vmax) vmax = db; }
      }
      return { S, frames, bins, hop, vmax };
    };
    const draw = () => {
      const { W, H } = size(); clear(ctx, W, H);
      const rect = { x: 55, y: 20, w: W - 75, h: H - 60 };
      const { S, frames, bins, hop, vmax } = compute();
      heatmap(ctx, S, rect, vmax - 70, vmax);
      axes(ctx, rect, { x0: 0, x1: LEN / FS, y0: 0, y1: FS / 2 }, { xlabel: "time (s)", ylabel: "frequency (Hz)", xfmt: (v) => v.toFixed(2), yfmt: (v) => v.toFixed(0) });
      out(`Window = <b>${win} samples</b> = <b>${(win / FS * 1000).toFixed(1)} ms</b> &nbsp;·&nbsp; hop = ${hop} (75% overlap) &nbsp;·&nbsp;
           frequency bins = win/2+1 = <b>${bins}</b> (<b>${(FS / win).toFixed(1)} Hz</b> per bin) &nbsp;·&nbsp; time frames = <b>${frames}</b><br>
           <span class="muted">${win <= 64 ? "Tiny window: the clicks at 0.30/0.31 s are razor sharp and separate, but the tone and chirp are thick smears — you can barely tell their frequencies."
          : win >= 1024 ? "Huge window: the 1200 Hz tone is a hairline and the chirp is crisp, but the two clicks 10 ms apart have merged into one wide blob — time is lost."
            : "Balanced: both the chirp/tone and the clicks are reasonably resolved. This is the region real speech pipelines live in (25–64 ms)."}</span>`);
    };
    const sel = h("select", { class: "w-select" });
    [32, 64, 128, 256, 512, 1024, 2048].forEach((v) => sel.appendChild(h("option", { value: v }, `${v} samples (${(v / FS * 1000).toFixed(1)} ms)`)));
    sel.value = win;
    sel.addEventListener("change", () => { win = parseInt(sel.value, 10); draw(); });
    controls.appendChild(h("label", { class: "w-slider" }, [h("span", { class: "w-label" }, "window size (n_fft)"), sel]));
    draw(); window.addEventListener("resize", draw);
  };

  /* ───────────────────────── 5. mel scale ───────────────────────── */
  WIDGETS["mel-scale"] = function (container) {
    const { controls, body } = frame(container, "Mel scale — how your ear warps the frequency axis",
      "Move the Hz slider and see where it lands on the mel axis. Then compare a +200 Hz step at low vs high frequency: same Hz, very different perceived change. The ticks below the curve show where n_mels filters would be centered — dense at the bottom, sparse at the top.");
    const { ctx, size } = makeCanvas(body, 280);
    const out = readout(body);
    const hz2mel = (f) => 2595 * Math.log10(1 + f / 700), mel2hz = (m) => 700 * (Math.pow(10, m / 2595) - 1);
    let fhz = 1000, nmels = 40;
    const FMAX = 8000;
    const draw = () => {
      const { W, H } = size(); clear(ctx, W, H);
      const rect = { x: 60, y: 20, w: W - 80, h: H - 80 };
      const range = { x0: 0, x1: FMAX, y0: 0, y1: hz2mel(FMAX) };
      axes(ctx, rect, range, { xlabel: "frequency (Hz)", ylabel: "mel", xticks: 8, xfmt: (v) => v.toFixed(0), yfmt: (v) => v.toFixed(0) });
      const xs = [], ys = [];
      for (let f = 0; f <= FMAX; f += 20) { xs.push(f); ys.push(hz2mel(f)); }
      plotLine(ctx, xs, ys, rect, range, PALETTE.blue, 2);
      // linear reference
      plotLine(ctx, [0, FMAX], [0, range.y1], rect, range, PALETTE.axis, 1, [4, 4]);
      // marker
      const m = hz2mel(fhz), px = rect.x + (fhz / FMAX) * rect.w, py = rect.y + rect.h - (m / range.y1) * rect.h;
      ctx.strokeStyle = PALETTE.orange; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(px, rect.y + rect.h); ctx.lineTo(px, py); ctx.lineTo(rect.x, py); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = PALETTE.orange; ctx.beginPath(); ctx.arc(px, py, 5, 0, 2 * Math.PI); ctx.fill();
      // mel filter centers as ticks along the bottom
      ctx.fillStyle = PALETTE.green;
      for (let i = 0; i <= nmels + 1; i++) {
        const fc = mel2hz((i / (nmels + 1)) * range.y1);
        const tx = rect.x + (fc / FMAX) * rect.w;
        ctx.fillRect(tx, rect.y + rect.h + 32, 1.5, 8);
      }
      ctx.fillStyle = PALETTE.text; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
      ctx.fillText(`${nmels} mel filter centers (Hz axis)`, rect.x, rect.y + rect.h + 52);
      const stepLo = hz2mel(fhz + 200) - m;
      out(`<b>${fhz} Hz → ${m.toFixed(0)} mel</b> &nbsp;·&nbsp; a +200 Hz step from here is worth <b>${stepLo.toFixed(0)} mel</b>
           (compare: 200→400 Hz = ${(hz2mel(400) - hz2mel(200)).toFixed(0)} mel, 6000→6200 Hz = ${(hz2mel(6200) - hz2mel(6000)).toFixed(0)} mel)<br>
           <span class="muted">Below ~1 kHz the curve is nearly linear (the ear is sharp); above it flattens (the ear is dull). Dashed grey = what a linear axis would look like.</span>`);
    };
    slider(controls, { label: "frequency", min: 0, max: FMAX, step: 50, value: fhz, fmt: fmt.hz }, (v) => { fhz = v; draw(); });
    slider(controls, { label: "n_mels", min: 8, max: 128, step: 8, value: nmels, fmt: fmt.int }, (v) => { nmels = v; draw(); });
    draw(); window.addEventListener("resize", draw);
  };

  /* ───────────────────────── 6. dB explorer ───────────────────────── */
  WIDGETS["db-explorer"] = function (container) {
    const { controls, body } = frame(container, "Decibels — why a log scale is the only way to see (and learn) audio",
      "Four sounds of very different loudness. On a linear scale everything except the loudest is invisible; in dB they are evenly spread. Drag the amplitude slider for the highlighted sound and watch both bars.");
    const { ctx, size } = makeCanvas(body, 220);
    const out = readout(body);
    const items = [{ n: "loud vowel", a: 1.0 }, { n: "normal speech", a: 0.1 }, { n: "soft consonant", a: 0.003 }, { n: "room noise", a: 0.0001 }];
    let user = 0.02;
    const draw = () => {
      const { W, H } = size(); clear(ctx, W, H);
      const half = (W - 100) / 2;
      const r1 = { x: 110, y: 20, w: half - 60, h: H - 50 }, r2 = { x: 110 + half + 20, y: 20, w: half - 60, h: H - 50 };
      title(ctx, "linear amplitude (0 → 1)", r1.x, 14); title(ctx, "decibels (−100 → 0 dB)", r2.x, 14);
      const all = items.concat([{ n: "your slider", a: user, hl: true }]);
      const rowH = r1.h / all.length;
      ctx.font = "11px sans-serif";
      all.forEach((it, i) => {
        const y = r1.y + i * rowH + 6, bh = rowH - 12;
        ctx.fillStyle = it.hl ? PALETTE.orange : PALETTE.text; ctx.textAlign = "right"; ctx.fillText(it.n, r1.x - 8, y + bh / 2 + 4);
        ctx.fillStyle = it.hl ? PALETTE.orange : PALETTE.blue;
        ctx.fillRect(r1.x, y, Math.max(1, it.a * r1.w), bh);
        const db = 20 * Math.log10(Math.max(it.a, 1e-5));
        ctx.fillStyle = it.hl ? PALETTE.orange : PALETTE.green;
        ctx.fillRect(r2.x, y, ((db + 100) / 100) * r2.w, bh);
        ctx.fillStyle = PALETTE.text; ctx.textAlign = "left";
        ctx.fillText(`${db.toFixed(1)} dB`, r2.x + ((db + 100) / 100) * r2.w + 6, y + bh / 2 + 4);
      });
      ctx.strokeStyle = PALETTE.axis; ctx.strokeRect(r1.x, r1.y, r1.w, r1.h); ctx.strokeRect(r2.x, r2.y, r2.w, r2.h);
      const db = 20 * Math.log10(Math.max(user, 1e-5));
      out(`amplitude <b>${user.toExponential(2)}</b> → <b>${db.toFixed(1)} dB</b> &nbsp;·&nbsp; rules of thumb: ×2 amplitude = +6 dB, ×10 = +20 dB, ×0.5 = −6 dB<br>
           <span class="muted">On the linear side, "soft consonant" and "room noise" are both a hairline — a model trained with MSE on linear values could not tell them apart. In dB they are 30 dB (~30% of the range) apart. Your ear works in dB; so should your loss function.</span>`);
    };
    slider(controls, { label: "your amplitude (log slider)", min: -5, max: 0, step: 0.05, value: Math.log10(user), fmt: (v) => Math.pow(10, v).toExponential(2) }, (v) => { user = Math.pow(10, v); draw(); });
    draw(); window.addEventListener("resize", draw);
  };

  /* ───────────────────────── 7. shape calculator ───────────────────────── */
  WIDGETS["stft-shapes"] = function (container) {
    const { controls, body } = frame(container, "Shape calculator — from parameters to tensor sizes",
      "Type any configuration and see every derived number: samples, frames, bins, milliseconds, Hz per bin. This is the arithmetic you will do in your head for the rest of your audio career.");
    const out = readout(body);
    const p = { sr: 16000, dur: 3.4, nfft: 1024, hop: 512, nmels: 128 };
    const num = (label, key, step) => {
      const inp = h("input", { type: "number", value: p[key], step, class: "w-num" });
      inp.addEventListener("input", () => { p[key] = parseFloat(inp.value) || 0; draw(); });
      controls.appendChild(h("label", { class: "w-slider" }, [h("span", { class: "w-label" }, label), inp]));
    };
    num("sample_rate", "sr", 1000); num("duration (s)", "dur", 0.1); num("n_fft", "nfft", 64); num("hop_length", "hop", 32); num("n_mels", "nmels", 8);
    const draw = () => {
      const samples = Math.round(p.sr * p.dur), bins = Math.floor(p.nfft / 2) + 1, frames = 1 + Math.floor(samples / p.hop);
      const ok = p.nmels <= bins;
      out(`<table class="w-table">
        <tr><td>samples</td><td><b>${samples.toLocaleString()}</b></td><td class="muted">sample_rate × duration</td></tr>
        <tr><td>Nyquist (max frequency)</td><td><b>${(p.sr / 2).toLocaleString()} Hz</b></td><td class="muted">sample_rate / 2</td></tr>
        <tr><td>window duration</td><td><b>${(p.nfft / p.sr * 1000).toFixed(1)} ms</b></td><td class="muted">n_fft / sample_rate</td></tr>
        <tr><td>hop duration</td><td><b>${(p.hop / p.sr * 1000).toFixed(1)} ms</b></td><td class="muted">hop_length / sample_rate &nbsp;(overlap ${p.nfft ? Math.round((1 - p.hop / p.nfft) * 100) : 0}%)</td></tr>
        <tr><td>linear frequency bins</td><td><b>${bins}</b></td><td class="muted">n_fft // 2 + 1</td></tr>
        <tr><td>Hz per linear bin</td><td><b>${(p.sr / p.nfft).toFixed(2)} Hz</b></td><td class="muted">sample_rate / n_fft</td></tr>
        <tr><td>time frames</td><td><b>${frames}</b></td><td class="muted">1 + samples // hop_length (center=True)</td></tr>
        <tr><td>linear spectrogram shape</td><td><b>(${bins}, ${frames})</b></td><td class="muted">bins × frames</td></tr>
        <tr><td>mel spectrogram shape</td><td><b>(${p.nmels}, ${frames})</b> ${ok ? "" : '<span class="bad">n_mels &gt; bins!</span>'}</td><td class="muted">n_mels × frames</td></tr>
        <tr><td>compression vs waveform</td><td><b>${(samples / (p.nmels * frames)).toFixed(1)}×</b> fewer numbers</td><td class="muted">samples / (n_mels × frames)</td></tr>
      </table>`);
    };
    draw();
  };

  /* ───────────────────────── mount ───────────────────────── */
  function mountAll() {
    document.querySelectorAll(".widget[data-widget]").forEach((el) => {
      const fn = WIDGETS[el.dataset.widget];
      if (fn) { try { fn(el); } catch (e) { el.innerHTML = `<p class="bad">widget error: ${e.message}</p>`; console.error(e); } }
      else el.innerHTML = `<p class="muted">unknown widget: ${el.dataset.widget}</p>`;
    });
  }
  window.NOTES_WIDGETS = WIDGETS;
  window.NOTES_FFT = { fft, magnitudeSpectrum };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mountAll); else mountAll();
})();
