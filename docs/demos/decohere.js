// "Measuring one side": joint momentum distribution P(p_A, p_B) of the pair
//   alpha |cat+>_A |L>_B + beta |cat->_A |R>_B
// as an apparatus of N pointer modes, coupled only to B, grows.
//
// Same parameters as entangled_collapse_sim.py (hbar = m = g = 1):
//   packet width s = 0.5, cat half-separation a = 2, L/R half-separation d = 2,
//   alpha^2 = 0.7, single-pointer width sigma = 6.
//
// The kick p_B -> p_B - P_tot is a convolution in p_B with a Gaussian of
// variance k = N / (4 sigma^2).  Before the kick
//   P0 = G(p_A) G(p_B) [ c0(p_A) + c1(p_A) sin(w p_B) ],   w = 2d,  G = exp(-p^2 / 2v),  v = 1/(4 s^2),
// and the convolution is done in closed form:
//   G * K           = r exp(-p^2 / 2(v+k))
//   (G sin(w.)) * K = r exp(-p^2 / 2(v+k)) exp(-w^2 v k / 2(v+k)) sin(w v p / (v+k)),   r = sqrt(v/(v+k)).
// The c1 term is the A-B interference; integrating over p_B kills it for every N,
// so A's momentum distribution never changes.

(function () {
  const cv = document.getElementById("decohere");
  if (!cv) return;
  const g = cv.getContext("2d");
  const W = cv.width, H = cv.height;

  const s = 0.5, a = 2, d = 2, alpha2 = 0.7, sigma = 6;
  const alpha = Math.sqrt(alpha2), beta = Math.sqrt(1 - alpha2);
  const v = 1 / (4 * s * s), w = 2 * d;
  const ov = Math.exp(-(a * a) / (2 * s * s));
  const np = 1 / Math.sqrt(2 * (1 + ov)), nm = 1 / Math.sqrt(2 * (1 - ov));

  const PMAX = 4, GRID = 140;
  const ps = Array.from({ length: GRID }, (_, i) => -PMAX + (2 * PMAX * (i + 0.5)) / GRID);
  const GA = ps.map((p) => Math.exp(-(p * p) / (2 * v)));
  const c0 = ps.map((p) => alpha2 * np * np * Math.cos(a * p) ** 2 + (1 - alpha2) * nm * nm * Math.sin(a * p) ** 2);
  const c1 = ps.map((p) => alpha * beta * np * nm * Math.sin(2 * a * p));

  // layout: heatmap with P(p_A) on the left and P(p_B) below
  const M = { l: 118, t: 14, size: 330 };
  const mx = M.l, my = M.t, ms = M.size;
  const sideW = 82, botH = 70;

  const off = document.createElement("canvas");
  off.width = off.height = GRID;
  const og = off.getContext("2d");
  const img = og.createImageData(GRID, GRID);

  // dark room -> laser -> amber -> cream
  const STOPS = [[0, [20, 16, 12]], [0.35, [150, 30, 24]], [0.6, [255, 59, 47]], [0.82, [238, 174, 94]], [1, [250, 238, 220]]];
  function ramp(t) {
    t = Math.max(0, Math.min(1, t));
    for (let i = 1; i < STOPS.length; i++) {
      if (t <= STOPS[i][0]) {
        const [t0, c0] = STOPS[i - 1], [t1, c1] = STOPS[i];
        const u = (t - t0) / (t1 - t0);
        return c0.map((c, j) => c + u * (c1[j] - c));
      }
    }
    return STOPS[STOPS.length - 1][1];
  }

  const slider = document.getElementById("decohere-n");
  const nOut = document.getElementById("decohere-nval");
  const cohOut = document.getElementById("decohere-coh");
  const cohBar = document.getElementById("decohere-bar");
  const playBtn = document.getElementById("decohere-play");

  const nFromSlider = () => Math.pow(2, +slider.value) - 1;

  function draw() {
    const N = nFromSlider();
    const k = N / (4 * sigma * sigma);
    const r = Math.sqrt(v / (v + k));
    const damp = Math.exp(-(w * w * v * k) / (2 * (v + k)));
    const env = ps.map((p) => r * Math.exp(-(p * p) / (2 * (v + k))));
    const fr = ps.map((p) => damp * Math.sin((w * v * p) / (v + k)));

    // P[iA][iB]
    let pmax = 0;
    const P = new Float64Array(GRID * GRID);
    const margB = new Float64Array(GRID);
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        const val = Math.max(0, GA[i] * env[j] * (c0[i] + c1[i] * fr[j]));
        P[i * GRID + j] = val;
        margB[j] += val;
        if (val > pmax) pmax = val;
      }
    }

    // heatmap (p_A up, p_B right)
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        const [R, G, B] = ramp(Math.sqrt(P[i * GRID + j] / pmax));
        const o = ((GRID - 1 - i) * GRID + j) * 4;
        img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = B; img.data[o + 3] = 255;
      }
    }
    og.putImageData(img, 0, 0);

    g.fillStyle = "#14100c";
    g.fillRect(0, 0, W, H);
    g.imageSmoothingEnabled = true;
    g.drawImage(off, mx, my, ms, ms);
    g.strokeStyle = "#3b3026";
    g.lineWidth = 1;
    g.strokeRect(mx + 0.5, my + 0.5, ms, ms);

    // axes labels
    g.fillStyle = "#b8a892";
    g.font = "italic 18px 'Source Serif 4', serif";
    g.textAlign = "center";
    g.fillText("p", mx + ms / 2 - 4, my + ms + botH + 26);
    g.font = "italic 13px 'Source Serif 4', serif";
    g.fillText("B", mx + ms / 2 + 6, my + ms + botH + 30);
    g.save();
    g.translate(18, my + ms / 2);
    g.rotate(-Math.PI / 2);
    g.font = "italic 18px 'Source Serif 4', serif";
    g.fillText("p", -4, 0);
    g.font = "italic 13px 'Source Serif 4', serif";
    g.fillText("A", 6, 4);
    g.restore();

    // A's marginal (left): the same for every N
    const margA = ps.map((_, i) => GA[i] * c0[i]);
    const maA = Math.max(...margA);
    g.fillStyle = "rgba(99,176,234,.18)";
    g.strokeStyle = "#63b0ea";
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(mx - 6, my + ms);
    for (let i = 0; i < GRID; i++) {
      const y = my + ms - ((i + 0.5) / GRID) * ms;
      g.lineTo(mx - 6 - (margA[i] / maA) * (sideW - 34), y);
    }
    g.lineTo(mx - 6, my);
    g.closePath();
    g.fill();
    g.stroke();

    // B's marginal (below): broadens as B is kicked
    let bmax0 = 0; // B's peak before the kick
    for (let i = 0; i < GRID; i++) bmax0 += GA[i] * c0[i];
    g.fillStyle = "rgba(238,174,94,.16)";
    g.strokeStyle = "#eeae5e";
    g.beginPath();
    g.moveTo(mx, my + ms + 6);
    for (let j = 0; j < GRID; j++) {
      const x = mx + ((j + 0.5) / GRID) * ms;
      g.lineTo(x, my + ms + 6 + (margB[j] / bmax0) * (botH - 12));
    }
    g.lineTo(mx + ms, my + ms + 6);
    g.closePath();
    g.fill();
    g.stroke();

    // side captions
    g.font = "600 12px Inter, sans-serif";
    g.textAlign = "left";
    g.fillStyle = "#63b0ea";
    g.fillText("A alone:", mx + ms + 14, my + 14);
    g.fillStyle = "#b8a892";
    g.font = "12px Inter, sans-serif";
    g.fillText("never changes", mx + ms + 14, my + 30);
    g.fillStyle = "#eeae5e";
    g.font = "600 12px Inter, sans-serif";
    g.fillText("B alone:", mx + ms + 14, my + ms + 26);
    g.fillStyle = "#b8a892";
    g.font = "12px Inter, sans-serif";
    g.fillText("kicked, spreads", mx + ms + 14, my + ms + 42);

    const coh = Math.exp(-(N * (2 * d) ** 2) / (8 * sigma * sigma));
    nOut.textContent = N < 0.5 ? "0" : String(Math.round(N));
    cohOut.textContent = coh.toFixed(3);
    cohBar.style.width = (coh * 100).toFixed(1) + "%";
  }

  slider.addEventListener("input", draw);

  let raf = null;
  function stop() {
    if (raf) cancelAnimationFrame(raf);
    raf = null;
    playBtn.textContent = "▶ Grow the apparatus";
  }
  playBtn.addEventListener("click", () => {
    if (raf) return stop();
    if (+slider.value >= +slider.max) slider.value = 0;
    playBtn.textContent = "❚❚ Pause";
    let last = null;
    const step = (now) => {
      if (last !== null) slider.value = Math.min(+slider.max, +slider.value + ((now - last) / 1000) * 1.1);
      last = now;
      draw();
      if (+slider.value >= +slider.max) return stop();
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
  });

  draw();
})();
