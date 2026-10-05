// "Follow a cloud, not a point": an ensemble of Newtonian particles in phase space.
// Harmonic V = x^2/2 rotates the cloud rigidly; quartic V = x^4/4 shears it into filaments.
// The strip on the right is the cloud's shadow on the x wall (a histogram of positions).

(function () {
  const cv = document.getElementById("cloud");
  if (!cv) return;
  const g = cv.getContext("2d");

  const N = 2500, L = 2.6;
  const PAD = { l: 30, r: 120, t: 8, b: 28 };
  const W = cv.width - PAD.l - PAD.r, H = cv.height - PAD.t - PAD.b;
  const xs = new Float64Array(N), pv = new Float64Array(N);
  let pot = "quartic";

  const force = { harmonic: (x) => -x, quartic: (x) => -x * x * x };

  function gauss() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function reset() {
    for (let i = 0; i < N; i++) { xs[i] = 1.6 + 0.22 * gauss(); pv[i] = 0.22 * gauss(); }
  }

  function step(dt) {
    const F = force[pot];
    for (let i = 0; i < N; i++) {          // leapfrog: kick-drift-kick, M = 1
      pv[i] += 0.5 * dt * F(xs[i]);
      xs[i] += dt * pv[i];
      pv[i] += 0.5 * dt * F(xs[i]);
    }
  }

  const sx = (p) => PAD.l + (p + L) / (2 * L) * W;   // p horizontal
  const sy = (x) => PAD.t + (L - x) / (2 * L) * H;   // x vertical, up = +x
  const NB = 70, hist = new Float64Array(NB);

  function draw() {
    g.fillStyle = "#14100c";
    g.fillRect(0, 0, cv.width, cv.height);
    g.strokeStyle = "rgba(244,235,223,.16)";
    g.beginPath();
    g.moveTo(sx(0), PAD.t); g.lineTo(sx(0), PAD.t + H);
    g.moveTo(PAD.l, sy(0)); g.lineTo(PAD.l + W, sy(0));
    g.stroke();
    g.strokeStyle = "#3b3026";
    g.strokeRect(PAD.l + .5, PAD.t + .5, W - 1, H - 1);

    g.fillStyle = "rgba(238,174,94,.55)";
    hist.fill(0);
    for (let i = 0; i < N; i++) {
      const X = sx(pv[i]), Y = sy(xs[i]);
      if (X > PAD.l && X < PAD.l + W && Y > PAD.t && Y < PAD.t + H) g.fillRect(X - 1, Y - 1, 2.2, 2.2);
      const b = Math.floor((L - xs[i]) / (2 * L) * NB);
      if (b >= 0 && b < NB) hist[b]++;
    }

    // wall
    const wx = PAD.l + W + 14, ww = PAD.r - 18;
    const grd = g.createLinearGradient(wx, 0, wx + ww, 0);
    grd.addColorStop(0, "#3a2c1f"); grd.addColorStop(1, "#5a4630");
    g.fillStyle = grd;
    g.fillRect(wx, PAD.t, ww, H);
    let hmax = 0;
    for (let b = 0; b < NB; b++) hmax = Math.max(hmax, hist[b]);
    g.fillStyle = "rgba(20,16,12,.6)";
    const bh = H / NB;
    for (let b = 0; b < NB; b++) g.fillRect(wx, PAD.t + b * bh, (hist[b] / Math.max(hmax, 1)) * (ww - 6), bh + .5);

    g.fillStyle = "#b8a892";
    g.font = "italic 17px 'Source Serif 4', serif";
    g.textAlign = "center";
    g.fillText("p", PAD.l + W / 2, cv.height - 6);
    g.fillText("x", 12, PAD.t + H / 2 + 5);
    g.font = "12px Inter, sans-serif";
    g.fillText("shadow on wall", wx + ww / 2, cv.height - 8);
  }

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const sub = 6;
    for (let k = 0; k < sub; k++) step(1.1 * dt / sub);
    draw();
    requestAnimationFrame(frame);
  }
  reset();
  requestAnimationFrame(frame);

  document.querySelectorAll("[data-pot]").forEach((b) => {
    b.classList.toggle("on", b.dataset.pot === pot);
    b.addEventListener("click", () => {
      pot = b.dataset.pot;
      reset();
      document.querySelectorAll("[data-pot]").forEach((o) => o.classList.toggle("on", o === b));
    });
  });
  document.getElementById("reset").addEventListener("click", reset);
})();
