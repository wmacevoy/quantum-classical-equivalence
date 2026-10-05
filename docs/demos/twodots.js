// "Two dots, one hand": one source steers two beams at opposite walls.
// Each cat sees a dot that wanders; together the dots move in perfect lockstep,
// with no signal passing between the walls.

(function () {
  const cv = document.getElementById("twodots");
  if (!cv) return;
  const g = cv.getContext("2d");
  const Wd = cv.width, Hd = cv.height;
  const wallW = 70, mid = Wd / 2, cy = Hd / 2;

  // a smooth, unpredictable-looking wander shared by both beams
  function wander(t) {
    return 0.55 * Math.sin(1.3 * t) + 0.3 * Math.sin(2.9 * t + 1.1) + 0.15 * Math.sin(5.3 * t + 2.3);
  }

  function dot(x, y) {
    const r = g.createRadialGradient(x, y, 0, x, y, 14);
    r.addColorStop(0, "rgba(255,120,100,1)");
    r.addColorStop(0.3, "rgba(255,59,47,.85)");
    r.addColorStop(1, "rgba(255,59,47,0)");
    g.fillStyle = r;
    g.beginPath(); g.arc(x, y, 14, 0, 2 * Math.PI); g.fill();
  }

  function frame(now) {
    const t = now / 1000;
    const y = cy + 0.38 * Hd * wander(t);
    g.fillStyle = "#14100c";
    g.fillRect(0, 0, Wd, Hd);

    for (const x0 of [0, Wd - wallW]) {
      const grd = g.createLinearGradient(x0, 0, x0 + wallW, 0);
      grd.addColorStop(0, x0 === 0 ? "#5a4630" : "#3a2c1f");
      grd.addColorStop(1, x0 === 0 ? "#3a2c1f" : "#5a4630");
      g.fillStyle = grd;
      g.fillRect(x0, 0, wallW, Hd);
    }

    const xl = wallW - 6, xr = Wd - wallW + 6;
    g.strokeStyle = "rgba(255,59,47,.28)";
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(mid, cy); g.lineTo(xl, y);
    g.moveTo(mid, cy); g.lineTo(xr, y);
    g.stroke();

    // the hand / source
    g.fillStyle = "#271f18";
    g.strokeStyle = "#eeae5e";
    g.lineWidth = 2;
    g.beginPath(); g.arc(mid, cy, 22, 0, 2 * Math.PI); g.fill(); g.stroke();
    g.fillStyle = "#eeae5e";
    g.font = "600 13px Inter, sans-serif";
    g.textAlign = "center";
    g.fillText("source", mid, cy + 46);

    dot(xl, y);
    dot(xr, y);

    g.fillStyle = "#b8a892";
    g.font = "13px Inter, sans-serif";
    g.fillText("cat A's wall", wallW / 2, Hd - 10);
    g.fillText("cat B's wall", Wd - wallW / 2, Hd - 10);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
