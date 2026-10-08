// Looking-Glass seminar deck — navigation, scaling, math rendering, notes.
// Slide order lives here; each slide is an ordinary page in this folder.

const SLIDES = [
  "index.html",
  "02-cat.html",
  "03-questions.html",
  "04-two-pictures.html",
  "05-liouville.html",
  "06-cloud.html",
  "07-tools.html",
  "08-transform.html",
  "09-glass.html",
  "10-error.html",
  "11-shadow.html",
  "12-negative.html",
  "13-point.html",
  "14-entangle.html",
  "15-bell.html",
  "16-pointer.html",
  "16b-decohere.html",
  "17-condition.html",
  "18-signed.html",
  "19-apparatus.html",
  "20-limits.html",
  "21-secret.html",
  "21b-exact.html",
  "21c-vdw.html",
  "21d-gate.html",
  "22-refs.html",
];

const DECK_TITLE = "The Looking-Glass Transform";

(function () {
  const file = location.pathname.split("/").pop() || "index.html";
  const idx = Math.max(0, SLIDES.indexOf(file));
  const stage = document.querySelector(".stage");

  function go(i) {
    if (i < 0 || i >= SLIDES.length || i === idx) return;
    const keep = document.body.classList.contains("show-notes") ? "#notes" : "";
    location.href = SLIDES[i] + keep;
  }

  // Scale the fixed 1280x720 stage to the window.
  function fit() {
    const s = Math.min(window.innerWidth / 1280, window.innerHeight / 720);
    stage.style.transform = "scale(" + s + ")";
  }
  window.addEventListener("resize", fit);
  fit();

  // Footer and progress bar (skipped on the title slide).
  const slide = stage.querySelector(".slide");
  if (idx > 0 && slide) {
    const foot = document.createElement("div");
    foot.className = "slide-footer";
    foot.innerHTML =
      '<span>' + DECK_TITLE + ' <span class="dot">&#9679;</span></span><span>' +
      (idx + 1) + " / " + SLIDES.length + "</span>";
    stage.appendChild(foot);
  }
  const bar = document.createElement("div");
  bar.className = "progress";
  bar.style.width = ((idx + 1) / SLIDES.length) * 100 + "%";
  stage.appendChild(bar);

  // Edge buttons for mouse and touch users.
  [["prev", "‹", -1], ["next", "›", 1]].forEach(([cls, label, d]) => {
    if ((d < 0 && idx === 0) || (d > 0 && idx === SLIDES.length - 1)) return;
    const b = document.createElement("button");
    b.className = "navbtn " + cls;
    b.textContent = label;
    b.setAttribute("aria-label", cls === "prev" ? "Previous slide" : "Next slide");
    b.addEventListener("click", () => go(idx + d));
    document.body.appendChild(b);
  });

  // Help overlay.
  const help = document.createElement("div");
  help.className = "help";
  help.innerHTML =
    "<kbd>→</kbd> <kbd>space</kbd> next &nbsp; <kbd>←</kbd> previous<br>" +
    "<kbd>Home</kbd> / <kbd>End</kbd> first / last<br>" +
    "<kbd>n</kbd> speaker notes &nbsp; <kbd>?</kbd> this help<br>" +
    "Full screen: use the browser's own (F11, or ⌃⌘F on a Mac)<br>so it survives page changes.";
  document.body.appendChild(help);

  if (location.hash === "#notes") document.body.classList.add("show-notes");

  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
    switch (e.key) {
      case "ArrowRight": case "PageDown": case " ": case "ArrowDown":
        e.preventDefault(); go(idx + 1); break;
      case "ArrowLeft": case "PageUp": case "ArrowUp": case "Backspace":
        e.preventDefault(); go(idx - 1); break;
      case "Home": go(0); break;
      case "End": go(SLIDES.length - 1); break;
      case "n": case "N":
        document.body.classList.toggle("show-notes");
        history.replaceState(null, "", document.body.classList.contains("show-notes") ? "#notes" : location.pathname);
        break;
      case "?": case "h":
        document.body.classList.toggle("show-help"); break;
      case "Escape":
        document.body.classList.remove("show-help"); break;
    }
  });

  // Swipe on touch screens.
  let x0 = null;
  document.addEventListener("touchstart", (e) => { x0 = e.touches[0].clientX; }, { passive: true });
  document.addEventListener("touchend", (e) => {
    if (x0 === null) return;
    const dx = e.changedTouches[0].clientX - x0;
    if (Math.abs(dx) > 60) go(idx + (dx < 0 ? 1 : -1));
    x0 = null;
  });

  // Math.
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  }

  // Buttons inside demos should not keep focus, so space and arrows still navigate.
  document.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => b.blur()));
})();
