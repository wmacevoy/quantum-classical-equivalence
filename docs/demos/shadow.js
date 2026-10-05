// "The Born rule is a shadow": Wigner functions of a few harmonic-oscillator states
// (hbar = M = omega = 1). In a quadratic potential the classical Liouville flow is exact,
// so time evolution is a rigid rotation of w in phase space. The wall shows the momentum
// marginal rho(x) = \int w dp, computed numerically; "measure" fires laser dots sampled from it.

(function () {
  const ps = document.getElementById("ps");
  const wall = document.getElementById("wall");
  if (!ps || !wall) return;
  const g = ps.getContext("2d");
  const gw = wall.getContext("2d");

  const L = 4.5;                       // plot range [-L, L] in both x and p
  const PAD = { l: 34, r: 8, t: 8, b: 30 };
  const W = ps.width - PAD.l - PAD.r;  // plot width  (p axis, horizontal)
  const H = ps.height - PAD.t - PAD.b; // plot height (x axis, vertical, up = +x)
  const INV_PI = 1 / Math.PI;
  const A = 2.2;                       // displacement for coherent and cat states

  const STATES = {
    coh: {
      label: "coherent",
      w: (x, p) => INV_PI * Math.exp(-(x - A) * (x - A) - p * p),
    },
    fock1: {
      label: "n = 1",
      w: (x, p) => { const r2 = x * x + p * p; return INV_PI * (2 * r2 - 1) * Math.exp(-r2); },
    },
    sup: {
      label: "|0⟩ + |1⟩",
      w: (x, p) => { const r2 = x * x + p * p; return INV_PI * (r2 + Math.SQRT2 * x) * Math.exp(-r2); },
    },
    cat: {
      label: "cat",
      w: (x, p) => {
        const n = 2 * Math.PI * (1 + Math.exp(-A * A));
        return (Math.exp(-(x - A) * (x - A) - p * p) + Math.exp(-(x + A) * (x + A) - p * p) +
                2 * Math.exp(-x * x - p * p) * Math.cos(2 * A * p)) / n;
      },
    },
  };

  let state = "cat";
  let t = 0;
  let playing = true;
  let dots = [];

  // w at time t: pull back along the harmonic flow (rigid rotation).
  function wt(x, p) {
    const c = Math.cos(t), s = Math.sin(t);
    return STATES[state].w(x * c - p * s, x * s + p * c);
  }

  // Diverging colour map whose zero is the room's background:
  // positive -> sunlight amber, negative -> blue. Full scale is the bound 1/pi.
  const BG = [20, 16, 12], POS = [238, 174, 94], POS2 = [255, 238, 205], NEG = [99, 176, 234], NEG2 = [205, 232, 255];
  function mix(a, b, f) { return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]; }
  function colour(v) {
    let f = Math.min(1, Math.abs(v) * Math.PI);
    f = Math.pow(f, 0.8);
    const [c1, c2] = v >= 0 ? [POS, POS2] : [NEG, NEG2];
    return f < 0.75 ? mix(BG, c1, f / 0.75) : mix(c1, c2, (f - 0.75) / 0.25);
  }

  const NG = 140;
  const off = document.createElement("canvas");
  off.width = NG; off.height = NG;
  const go = off.getContext("2d");
  const img = go.createImageData(NG, NG);

  const NX = 220, NP = 320, PMAX = 6.5;
  const rho = new Float64Array(NX);

  function xOfRow(i) { return L - (2 * L) * (i + 0.5) / NX; }   // row 0 at top (+x)
  function yOfX(x) { return PAD.t + (L - x) / (2 * L) * H; }

  function compute() {
    let wmin = Infinity;
    for (let j = 0; j < NG; j++) {
      const x = L - 2 * L * (j + 0.5) / NG;
      for (let i = 0; i < NG; i++) {
        const p = -L + 2 * L * (i + 0.5) / NG;
        const v = wt(x, p);
        if (v < wmin) wmin = v;
        const c = colour(v);
        const k = 4 * (j * NG + i);
        img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = 255;
      }
    }
    go.putImageData(img, 0, 0);

    // The shadow: integrate out p.
    const dp = 2 * PMAX / NP;
    let rmin = Infinity;
    for (let i = 0; i < NX; i++) {
      const x = xOfRow(i);
      let s = 0;
      for (let k = 0; k < NP; k++) s += wt(x, -PMAX + (k + 0.5) * dp);
      rho[i] = s * dp;
      if (rho[i] < rmin) rmin = rho[i];
    }
    return { wmin, rmin };
  }

  function drawPhase() {
    g.fillStyle = "#14100c";
    g.fillRect(0, 0, ps.width, ps.height);
    g.imageSmoothingEnabled = true;
    g.drawImage(off, PAD.l, PAD.t, W, H);
    g.strokeStyle = "rgba(244,235,223,.18)";
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(PAD.l + W / 2, PAD.t); g.lineTo(PAD.l + W / 2, PAD.t + H);
    g.moveTo(PAD.l, PAD.t + H / 2); g.lineTo(PAD.l + W, PAD.t + H / 2);
    g.stroke();
    g.strokeStyle = "#3b3026";
    g.strokeRect(PAD.l + .5, PAD.t + .5, W - 1, H - 1);
    g.fillStyle = "#b8a892";
    g.font = "italic 17px 'Source Serif 4', serif";
    g.textAlign = "center";
    g.fillText("p", PAD.l + W / 2, ps.height - 8);
    g.fillText("x", 14, PAD.t + H / 2 + 5);
    g.font = "12px Inter, sans-serif";
    g.fillText("−" + L, PAD.l + 12, ps.height - 10);
    g.fillText("+" + L, PAD.l + W - 14, ps.height - 10);
  }

  function drawWall(now) {
    const ww = wall.width;
    // sunlit plaster
    const grd = gw.createLinearGradient(0, 0, ww, 0);
    grd.addColorStop(0, "#3a2c1f");
    grd.addColorStop(1, "#5a4630");
    gw.fillStyle = grd;
    gw.fillRect(0, 0, ww, wall.height);
    gw.fillStyle = "rgba(20,16,12,.55)";
    gw.fillRect(0, 0, ww, PAD.t);
    gw.fillRect(0, PAD.t + H, ww, wall.height - PAD.t - H);

    // shadow curve rho(x): horizontal extent from the left edge
    const SCALE = (ww - 16) / 0.62;
    gw.beginPath();
    gw.moveTo(0, yOfX(L));
    for (let i = 0; i < NX; i++) gw.lineTo(Math.max(0, rho[i]) * SCALE, yOfX(xOfRow(i)));
    gw.lineTo(0, yOfX(-L));
    gw.closePath();
    gw.fillStyle = "rgba(20,16,12,.55)";
    gw.fill();
    gw.strokeStyle = "#f6d3a2";
    gw.lineWidth = 2;
    gw.stroke();

    // laser dots, fading
    dots = dots.filter((d) => now - d.t0 < 4000);
    for (const d of dots) {
      const a = 1 - (now - d.t0) / 4000;
      const y = yOfX(d.x);
      const r = gw.createRadialGradient(d.u * ww, y, 0, d.u * ww, y, 9);
      r.addColorStop(0, "rgba(255,90,70," + a + ")");
      r.addColorStop(0.35, "rgba(255,59,47," + 0.75 * a + ")");
      r.addColorStop(1, "rgba(255,59,47,0)");
      gw.fillStyle = r;
      gw.beginPath(); gw.arc(d.u * ww, y, 9, 0, 2 * Math.PI); gw.fill();
    }

    gw.fillStyle = "#b8a892";
    gw.font = "13px Inter, sans-serif";
    gw.textAlign = "center";
    gw.fillText("wall: ∫ w dp", ww / 2, wall.height - 10);
  }

  function sample(n, now) {
    const cdf = new Float64Array(NX);
    let s = 0;
    for (let i = 0; i < NX; i++) { s += Math.max(0, rho[i]); cdf[i] = s; }
    for (let k = 0; k < n; k++) {
      const u = Math.random() * s;
      let lo = 0, hi = NX - 1;
      while (lo < hi) { const m = (lo + hi) >> 1; if (cdf[m] < u) lo = m + 1; else hi = m; }
      const x = xOfRow(lo) + (Math.random() - 0.5) * (2 * L / NX);
      dots.push({ x, u: 0.1 + 0.8 * Math.random(), t0: now - Math.random() * 300 * (n > 1) });
    }
  }

  const outW = document.getElementById("wmin");
  const outR = document.getElementById("rmin");
  let last = performance.now();
  let stats = compute();

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (playing) { t += 0.6 * dt; stats = compute(); }
    drawPhase();
    drawWall(now);
    if (outW) outW.textContent = stats.wmin.toFixed(3);
    if (outR) outR.textContent = Math.abs(stats.rmin) < 5e-4 ? "0.000" : stats.rmin.toFixed(3);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  document.querySelectorAll("[data-state]").forEach((b) => {
    if (b.dataset.state === state) b.classList.add("on");
    b.addEventListener("click", () => {
      state = b.dataset.state;
      t = 0; dots = [];
      stats = compute();
      document.querySelectorAll("[data-state]").forEach((o) => o.classList.toggle("on", o === b));
    });
  });
  const play = document.getElementById("play");
  play.addEventListener("click", () => {
    playing = !playing;
    play.textContent = playing ? "Pause" : "Play";
  });
  document.getElementById("m1").addEventListener("click", () => sample(1, performance.now()));
  document.getElementById("m200").addEventListener("click", () => sample(300, performance.now()));
})();
