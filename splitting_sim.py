#!/usr/bin/env python3
"""
Operator splitting for the exact dynamics w_t = C w + dQ w in V = x^4/4.

C is the whole classical Liouville operator (free flight + classical force);
dQ is the exact quantum correction, which in the p-hat picture is the pointwise
multiplier (i/hbar) dQ-hat with dQ-hat = V(x + p-hat/2) - V(x - p-hat/2) - p-hat V'(x)
= x p-hat^3 / 4 (the code scales p-hat by 1/hbar).  Two Strang schemes with step h over time T = n h:

  A. drift / kick / drift:  [e^{hD/2} e^{h(K + dQ)} e^{hD/2}]^n
     (the classical flow is itself split, so even a classical state has error [D, K]);
  B. classical / correction:  e^{hC/2} (e^{h dQ} e^{hC})^{n-1} e^{h dQ} e^{hC/2}
     with e^{hC} solved accurately as classical mechanics (here: leapfrog substeps of
     T0/1600, standing in for any accurate trajectory integrator) and e^{h dQ}
     applied exactly as e^{i h dQ-hat / hbar} in p-hat.

In B the only splitting error is the commutator [C, dQ], which is small where the
correction is small.  Each scheme is compared at t = 2 T0 with exact quantum
evolution in the eigenbasis (M = 1): the ground state and a displaced Gaussian
at hbar = 1, where the correction is of order one, and a coherent state at
hbar = 0.1, where it is small.

Outputs (in splitting_sim_out/): error_vs_h.png, summary.txt
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.linalg import eigh_tridiagonal
from scipy.interpolate import CubicSpline

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splitting_sim_out")
os.makedirs(OUT, exist_ok=True)

T0 = 6.5111
T_END = 2 * T0
SUB = T0 / 1600                       # classical substep inside e^{hC}
HS = T0 / np.array([800, 400, 200, 100, 50, 25, 12.5])


def V(z):
    return z**4 / 4


class Grid:
    """Phase-space grid and split-step factors for a given hbar.  p-hat here is the
    variable conjugate to p, so dQ-hat / hbar = hbar^2 x p-hat^3 / 4."""

    def __init__(self, hbar, XL, PL, N):
        self.hbar = hbar
        self.x = -XL + 2 * XL * np.arange(N) / N
        self.p = -PL + 2 * PL * np.arange(N) / N
        self.dx, self.dp = self.x[1] - self.x[0], self.p[1] - self.p[0]
        self.kx = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)
        ph = 2 * np.pi * np.fft.fftfreq(N, d=self.dp)
        self.X, self.PH = np.meshgrid(self.x, ph, indexing="ij")

    def drift(self, w, dt):
        return np.fft.ifft(np.fft.fft(w, axis=0) * np.exp(-1j * np.outer(self.kx, self.p) * dt), axis=0).real

    @staticmethod
    def mult_p(w, factor):
        return np.fft.ifft(np.fft.fft(w, axis=1) * factor, axis=1).real

    def kick(self, dt):
        return np.exp(1j * dt * self.PH * self.X**3)                       # classical force

    def dq(self, dt):
        return np.exp(1j * dt * self.hbar**2 * self.X * self.PH**3 / 4)     # exact correction

    def classical(self, w, h):
        """e^{hC}: accurate classical flow by leapfrog substeps of about SUB."""
        m = max(1, int(round(h / SUB)))
        s = h / m
        K = self.kick(s)
        w = self.drift(w, s / 2)
        for _ in range(m - 1):
            w = self.drift(self.mult_p(w, K), s)
        return self.drift(self.mult_p(w, K), s / 2)

    def scheme_a(self, w, h, n):
        F = self.kick(h) * self.dq(h)
        for _ in range(n):
            w = self.drift(self.mult_p(self.drift(w, h / 2), F), h / 2)
        return w

    def scheme_b(self, w, h, n):
        Q = self.dq(h)
        w = self.classical(w, h / 2)
        for _ in range(n - 1):
            w = self.classical(self.mult_p(w, Q), h)
        return self.classical(self.mult_p(w, Q), h / 2)


# ---------------------------------------------------------------- exact quantum reference
def eigenbasis(hbar, xg):
    hg = xg[1] - xg[0]
    E, Psi = eigh_tridiagonal(hbar**2 / hg**2 + V(xg), -0.5 * hbar**2 / hg**2 * np.ones(xg.size - 1),
                              select="i", select_range=(0, 150))
    return E, Psi / np.sqrt(hg)


def wigner(g, xg, psi_x):
    """w = (1/(pi hbar)) int dy psi*(x+y) psi(x-y) e^{2ipy/hbar} on the grid g."""
    re, im = CubicSpline(xg, psi_x.real), CubicSpline(xg, psi_x.imag)
    y = np.linspace(0, xg[-1] - g.x[-1], 2001)
    wy = np.full(y.size, y[1] - y[0]); wy[[0, -1]] /= 2
    out = np.zeros((g.x.size, g.p.size))
    E2 = np.exp(2j * np.outer(g.p, y) / g.hbar)
    for i, xv in enumerate(g.x):
        a1 = re(xv + y) + 1j * im(xv + y)
        a2 = re(xv - y) + 1j * im(xv - y)
        out[i] = 2 * np.real(E2 @ (np.conj(a1) * a2 * wy / (np.pi * g.hbar)))
    return out


def gaussian(xg, x0, s):
    return (np.pi * s**2) ** -0.25 * np.exp(-(xg - x0) ** 2 / (2 * s**2)) + 0j


# (name, hbar, XL, PL, grid points per axis, reference grid, initial state)
xg1 = np.linspace(-9, 9, 14401)
xg2 = np.linspace(-6, 6, 19201)
cases = [
    ("ground state, hbar = 1", 1.0, 6.0, 8.0, 512, xg1, None),
    ("displaced Gaussian x0 = 1.5, hbar = 1", 1.0, 6.0, 8.0, 512, xg1, lambda xg: gaussian(xg, 1.5, 0.6)),
    ("coherent state x0 = 1.5, hbar = 0.1", 0.1, 3.5, 3.5, 768, xg2, lambda xg: gaussian(xg, 1.5, np.sqrt(0.1))),
]
results = {}
for name, hbar, XL, PL, N, xg, init in cases:
    g = Grid(hbar, XL, PL, N)
    E, Psi = eigenbasis(hbar, xg)
    hg = xg[1] - xg[0]
    psi = Psi[:, 0].astype(complex) if init is None else init(xg)
    c = Psi.T @ psi * hg
    w0 = wigner(g, xg, psi)
    psiT = Psi @ (c * np.exp(-1j * E * T_END / hbar))
    wq = wigner(g, xg, psiT)
    bq = np.interp(g.x, xg, np.abs(psiT) ** 2)
    errs = {"A": [], "B": []}
    for h in HS:
        n = int(round(T_END / h))
        for lab, sch in (("A", g.scheme_a), ("B", g.scheme_b)):
            w = sch(w0.copy(), T_END / n, n)
            errs[lab].append((np.abs(w - wq).max() / np.abs(wq).max(),
                              0.5 * np.abs(w.sum(axis=1) * g.dp - bq).sum() * g.dx))
        print(f"{name}: h = T0/{T0 / h:g} done", flush=True)
    results[name] = {k: np.array(v) for k, v in errs.items()}
    results[name]["captured"] = float(np.sum(np.abs(c) ** 2))
    results[name]["N"] = N

lines = ["Operator splitting in V = x^4/4 (M = 1), error at t = 2 T0 vs exact quantum",
         "A: drift/kick+correction/drift Strang.  B: e^{hC/2}(e^{h dQ} e^{hC})^{n-1} e^{h dQ} e^{hC/2},",
         f"   e^{{hC}} by classical substeps of T0/{T0 / SUB:g}.", ""]
for name, r in results.items():
    lines.append(f"{name}   (eigenbasis captures {r['captured']:.8f}, grid {r['N']}^2)")
    lines.append("     h        A: max|dw|/max|w|  Born TV     B: max|dw|/max|w|  Born TV")
    for i, h in enumerate(HS):
        lines.append(f"   T0/{T0 / h:<6g} {r['A'][i, 0]:14.2e} {r['A'][i, 1]:10.2e}"
                     f" {r['B'][i, 0]:18.2e} {r['B'][i, 1]:10.2e}")
    lines.append("")
summary = "\n".join(lines)
print(summary)
open(os.path.join(OUT, "summary.txt"), "w").write(summary + "\n")

fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
for ax, (name, r) in zip(axes, results.items()):
    ax.loglog(T0 / HS, r["A"][:, 0], "o-", label="A: drift / kick + correction / drift")
    ax.loglog(T0 / HS, r["B"][:, 0], "s-", label="B: classical flow / correction")
    ax.set_xlabel("steps per period  T$_0$/h"); ax.set_ylabel("max|w - w$_q$| / max|w$_q$|  at t = 2T$_0$")
    ax.set_title(name, fontsize=10)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].legend(fontsize=8)
fig.savefig(os.path.join(OUT, "error_vs_h.png"), dpi=130)
