#!/usr/bin/env python3
"""
Slide figures for the looking-glass potential (docs/img/lj_glass.png, docs/img/transmon_glass.png).

Lennard-Jones: exact segment average of 4eps[(s/r)^12 - (s/r)^6] at the thermal coherence
length p_hat = sigma Lambda*/(2 pi) (kT = eps; de Boer parameters Ar 0.186, He 2.64).
Transmon: the segment average of -E_J cos(phi) is -E_J sinc(p_hat/2) cos(phi) exactly;
levels from diagonalizing H = 4 E_C n^2 - E_J cos(phi) in the charge basis (E_J/E_C = 50).
Prints the numbers quoted on slides 24-25.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BG, INK, MUTED, AMBER, LASER, BLUE, GREEN, RULE = "#14100c", "#f4ebdf", "#b8a892", "#eeae5e", "#ff3b2f", "#63b0ea", "#8fcf8a", "#3b3026"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": RULE, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.size": 15, "axes.titlesize": 16, "legend.frameon": False})
# ---------------- Lennard-Jones (reduced units r/sigma, V/eps)
V = lambda r: 4 * (r**-12 - r**-6)
def VLG(r, a):   # exact segment average of 4(r^-12 - r^-6) over [r-a/2, r+a/2]
    lo, hi = r - a / 2, r + a / 2
    return 4 / a * ((lo**-11 - hi**-11) / 11 - (lo**-5 - hi**-5) / 5)
Lam = {"Ar": 0.186, "He": 2.64}               # de Boer parameters
ah = {k: v / (2 * np.pi) for k, v in Lam.items()}   # coherence length / sigma at kT = eps
rm = 2 ** (1 / 6)
print("LJ: thermal coherence length p_hat/sigma at kT=eps:", ah)
for k, a in ah.items():
    print(f"  {k}: dV(r_m)/eps exact = {VLG(rm, a) - V(rm):.4f}, leading p^2 V''/24 = {a**2 * 72 / 2**(1/3) / 24:.4f}, "
          f"well depth V_LG(min) = {min(VLG(np.linspace(1.0, 2, 4000), a)):.4f}")
r = np.linspace(0.88, 2.6, 800)
fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
ax.plot(r, V(r), color=AMBER, lw=3, label="bare $V$")
ax.plot(r, VLG(r, ah["Ar"]), color=GREEN, lw=2, ls="--", label=r"argon, $\hat p \approx 0.03\sigma$")
ok = r - ah["He"] / 2 > 0.6
ax.plot(r[ok], VLG(r[ok], ah["He"]), color=LASER, lw=2.5, label=r"helium, $\hat p \approx 0.42\sigma$")
ax.axhline(0, color=RULE, lw=1)
ax.set_ylim(-1.25, 1.5); ax.set_xlim(0.88, 2.6)
ax.set_xlabel(r"$r/\sigma$"); ax.set_ylabel(r"$V/\varepsilon$"); ax.legend(loc="upper right")
ax.set_title(r"Lennard-Jones: bare vs looking-glass, $k_BT=\varepsilon$")
fig.savefig("docs/img/lj_glass.png", dpi=150)
# ---------------- transmon
EC, EJ = 1.0, 50.0
n = np.arange(-30, 31)
H = np.diag(4 * EC * n**2.0) - EJ / 2 * (np.eye(61, k=1) + np.eye(61, k=-1))
E = np.linalg.eigvalsh(H)
w01, w12 = E[1] - E[0], E[2] - E[1]
sphi = (2 * EC / EJ) ** 0.25
print(f"transmon EJ/EC=50: w01 = {w01:.3f} EC, w12 = {w12:.3f} EC, alpha = {w12 - w01:.3f} EC, |alpha|/w01 = {abs(w12 - w01) / w01:.4f}")
print(f"  sigma_phi = {sphi:.3f}, p_hat rms = 2 sigma = {2 * sphi:.3f}, 1 - sinc(p_hat/2) at p_hat = 2 sigma: {1 - np.sinc(sphi / np.pi):.4f}")
phi = np.linspace(-np.pi, np.pi, 600)
fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
ax.plot(phi, -np.cos(phi), color=AMBER, lw=3, label=r"bare $-E_J\cos\varphi$")
for ph, c, lab in [(2 * sphi, GREEN, r"$\hat p = 2\sigma_\varphi$ (qubit)"), (np.pi, LASER, r"$\hat p = \pi$")]:
    ax.plot(phi, -np.sinc(ph / 2 / np.pi) * np.cos(phi), color=c, lw=2.2, ls="--" if c == GREEN else "-", label=lab)
for k in range(4):
    y = E[k] / EJ; ft = np.arccos(-y)
    ax.hlines(y, -ft, ft, color=BLUE, lw=1.4, alpha=0.9)
ax.text(np.arccos(-E[1] / EJ) + 0.08, E[1] / EJ, r"$|1\rangle$", color=BLUE, va="center", fontsize=13)
ax.text(np.arccos(-E[0] / EJ) + 0.08, E[0] / EJ, r"$|0\rangle$", color=BLUE, va="center", fontsize=13)
ax.set_xlabel(r"phase $\varphi$"); ax.set_ylabel(r"$V/E_J$")
ax.set_ylim(-1.1, 1.1); ax.set_xlim(-np.pi, np.pi); ax.legend(loc="upper center", fontsize=12)
ax.set_title(r"Transmon, $E_J/E_C = 50$: $\hat V = -E_J\,\mathrm{sinc}(\hat p/2)\cos\varphi$")
fig.savefig("docs/img/transmon_glass.png", dpi=150)
