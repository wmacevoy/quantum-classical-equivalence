#!/usr/bin/env python3
"""
Classical trajectories plus signed momentum jumps: an exact form of quantum
dynamics in V = x^4/4.

The exact (Schrodinger/Wigner) dynamics of w is the classical flux law with one
extra term, the segment-averaged correction to the force.  In the p-hat
picture (p-hat conjugate to p) one time step factors as

    drift : w(x, p) -> w(x - p dt/M, p)                 classical free flight
    kick  : w^(x, p-hat) -> e^{ i dt p-hat V'(x) } w^       classical force, p -> p - V' dt
    jump  : w^(x, p-hat) -> e^{ i dt x p-hat^3 / 4 } w^     segment-averaged correction (hbar = 1)

The kick and the jump together are exactly e^{ i dt [V(x+p-hat/2) - V(x-p-hat/2)] },
because V(x+p-hat/2) - V(x-p-hat/2) = p-hat x^3 + x p-hat^3/4 for the quartic well.
Without the jump factor the scheme is classical Liouville flow; with it, it is
quantum dynamics.  In p the jump factor is a real, signed kernel K(q): the
correction does not move phase-space points but redistributes signed weight
across momentum transfers q.

Why deterministic.  The jump kernel can be sampled as a signed-particle Monte
Carlo (Nedjalkov et al., Phys. Rev. B 70, 115319 (2004)): particles fly
classically and spawn +/- pairs at p +/- q.  Tried here with a coherence cutoff
L = 4 and cell annihilation, it suffers the sign problem: the total |weight|
grew from 1.2 to 5 in 50 steps (0.8 time units) and to 5e5 by 200 steps, and
coarser annihilation cells made it worse.  Applying the same jump kernel
deterministically on a phase-space grid has no sign problem, and is what this
script does.

Tests (hbar = M = 1), each against exact quantum evolution in the eigenbasis:
  1. the ground state of x^4/4 (exact answer: stationary);
  2. a displaced Gaussian (width 0.6, centre x = 1.5).

Outputs (in signed_particle_sim_out/): drift_vs_t.png, born.png, wigner.png,
kernel.png, summary.txt
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.interpolate import CubicSpline

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signed_particle_sim_out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- parameters
NX, XL = 512, 6.0       # x grid on [-XL, XL)
NP, PL = 512, 8.0       # p grid on [-PL, PL)
T0 = 6.5111             # classical period of the ground-state orbit
DT = T0 / 800
PERIODS = 6.0
NSAMP = 61

x = -XL + 2 * XL * np.arange(NX) / NX
p = -PL + 2 * PL * np.arange(NP) / NP
dx, dp = x[1] - x[0], p[1] - p[0]
kx = 2 * np.pi * np.fft.fftfreq(NX, d=dx)          # conjugate to x
ph = 2 * np.pi * np.fft.fftfreq(NP, d=dp)          # p-hat, conjugate to p
X, PH = np.meshgrid(x, ph, indexing="ij")
DRIFT = np.exp(-1j * np.outer(kx, p) * DT / 2)     # half drift: w(x - p dt/2, p)
KICK = np.exp(1j * DT * PH * X**3)                 # classical kick: w(x, p + V' dt)
JUMP = np.exp(1j * DT * X * PH**3 / 4)             # segment-averaged correction


def V(z):
    return z**4 / 4


def step(w, jumps):
    """Strang step: half drift, kick (+ jump), half drift."""
    w = np.fft.ifft(np.fft.fft(w, axis=0) * DRIFT, axis=0).real
    factor = KICK * JUMP if jumps else KICK
    w = np.fft.ifft(np.fft.fft(w, axis=1) * factor, axis=1).real
    return np.fft.ifft(np.fft.fft(w, axis=0) * DRIFT, axis=0).real


# ---------------------------------------------------------------- exact quantum reference
xg = np.linspace(-9, 9, 3601)
hg = xg[1] - xg[0]
E, Psi = eigh_tridiagonal(1 / hg**2 + V(xg), -0.5 / hg**2 * np.ones(xg.size - 1),
                          select="i", select_range=(0, 150))
Psi /= np.sqrt(hg)


def wigner(psi_x):
    """w = (1/pi) int dy psi*(x+y) psi(x-y) e^{2ipy} on the (x, p) grid."""
    re, im = CubicSpline(xg, psi_x.real), CubicSpline(xg, psi_x.imag)
    y = np.linspace(0, 7, 1401)
    wy = np.full(y.size, y[1] - y[0]); wy[[0, -1]] /= 2
    out = np.zeros((NX, NP))
    E2 = np.exp(2j * np.outer(p, y))
    for i, xv in enumerate(x):
        a1 = re(xv + y) + 1j * im(xv + y)
        a2 = re(xv - y) + 1j * im(xv - y)
        out[i] = 2 * np.real(E2 @ (np.conj(a1) * a2 * wy / np.pi))
    return out


def born_grid(w):
    return w.sum(axis=1) * dp


cases = {
    "ground state": Psi[:, 0].astype(complex),
    "displaced Gaussian": (np.pi * 0.36) ** -0.25 * np.exp(-(xg - 1.5) ** 2 / (2 * 0.36)) + 0j,
}
ts = np.linspace(0, PERIODS * T0, NSAMP)
sample_steps = np.round(ts / DT).astype(int)

results = {}
for name, psi in cases.items():
    c = Psi.T @ psi * hg
    captured = float(np.sum(np.abs(c) ** 2))
    w0 = wigner(psi)
    runs = {}
    for label, jumps in (("classical", False), ("trajectories + jumps", True)):
        w = w0.copy()
        snaps, k = [], 0
        for n in range(sample_steps[-1] + 1):
            if n == sample_steps[k]:
                snaps.append(w.copy())
                k += 1
                if k == len(sample_steps):
                    break
            w = step(w, jumps)
        runs[label] = snaps
    born_q = []
    for t in ts:
        psit = Psi @ (c * np.exp(-1j * E * t))
        born_q.append(np.interp(x, xg, np.abs(psit) ** 2))
    tv = {lab: np.array([0.5 * np.sum(np.abs(born_grid(s) - bq)) * dx for s, bq in zip(sn, born_q)])
          for lab, sn in runs.items()}
    wq_end = wigner(Psi @ (c * np.exp(-1j * E * ts[-1])))
    werr = {lab: np.abs(sn[-1] - wq_end).max() / np.abs(wq_end).max() for lab, sn in runs.items()}
    results[name] = dict(runs=runs, tv=tv, born_q=born_q, wq_end=wq_end, werr=werr,
                         captured=captured, norm0=w0.sum() * dx * dp)
    print(f"{name}: done (captured |c|^2 = {captured:.6f})", flush=True)

# ---------------------------------------------------------------- summary
lines = ["Classical trajectories + signed momentum jumps vs exact quantum, V = x^4/4 (hbar = M = 1)",
         f"grid {NX} x {NP} on [-{XL},{XL}) x [-{PL},{PL}),  dt = T0/800,  T0 = {T0}",
         ""]
for name, r in results.items():
    lines.append(f"{name}: Born-density TV distance from exact quantum   (int w0 = {r['norm0']:.5f})")
    lines.append("     t/T0    classical   trajectories+jumps")
    for i in range(0, NSAMP, 10):
        lines.append(f"   {ts[i] / T0:6.2f}   {r['tv']['classical'][i]:9.5f}   {r['tv']['trajectories + jumps'][i]:12.2e}")
    lines.append(f"   max |w - w_quantum| / max |w_quantum| at t = {PERIODS:g} T0:  classical "
                 f"{r['werr']['classical']:.3f},  trajectories+jumps {r['werr']['trajectories + jumps']:.2e}")
    lines.append("")
lines += ["Stochastic signed particles (same kernel, coherence cutoff L = 4, cell annihilation):",
          "sign problem; total |weight| 1.2 -> 5 (50 steps) -> 5e5 (200 steps). Not used."]
summary = "\n".join(lines)
print(summary)
open(os.path.join(OUT, "summary.txt"), "w").write(summary + "\n")

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for ax, (name, r) in zip(axes, results.items()):
    ax.semilogy(ts / T0, np.maximum(r["tv"]["classical"], 1e-16), label="classical flow")
    ax.semilogy(ts / T0, np.maximum(r["tv"]["trajectories + jumps"], 1e-16), label="trajectories + jumps")
    ax.set_title(name); ax.set_xlabel("t / T$_0$"); ax.set_ylabel("Born TV distance from exact quantum")
axes[0].legend()
fig.savefig(os.path.join(OUT, "drift_vs_t.png"), dpi=130)

fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
for row, (name, r) in enumerate(results.items()):
    for col, i in enumerate([NSAMP // 6, NSAMP // 2, NSAMP - 1]):
        ax = axes[row, col]
        ax.plot(x, r["born_q"][i], "k", lw=2.5, label="exact quantum")
        ax.plot(x, born_grid(r["runs"]["classical"][i]), "C0--", label="classical flow")
        ax.plot(x, born_grid(r["runs"]["trajectories + jumps"][i]), "C1", lw=1, label="trajectories + jumps")
        ax.set_xlim(-3.5, 3.5); ax.set_title(f"{name}, t = {ts[i] / T0:.1f} T$_0$", fontsize=9)
        ax.set_xlabel("x")
axes[0, 0].legend(fontsize=8)
fig.savefig(os.path.join(OUT, "born.png"), dpi=130)

r = results["displaced Gaussian"]
vm = np.abs(r["wq_end"]).max()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
for ax, (lab, w) in zip(axes, [("exact quantum", r["wq_end"]), ("classical flow", r["runs"]["classical"][-1]),
                               ("trajectories + jumps", r["runs"]["trajectories + jumps"][-1])]):
    ax.imshow(w.T, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm, extent=[x[0], x[-1], p[0], p[-1]], aspect="auto")
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-4, 4); ax.set_title(f"{lab}, t = {PERIODS:g} T$_0$", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("p")
fig.savefig(os.path.join(OUT, "wigner.png"), dpi=130)

# the jump kernel at x = 1 over T0/50: signed weight moved by momentum transfer q
nq, dq = 4096, 0.005
phq = 2 * np.pi * np.fft.fftfreq(nq, d=dq)
K = np.fft.fftshift(np.fft.ifft(np.exp(1j * T0 / 50 * phq**3 * np.exp(-(phq / 6) ** 8) / 4) - 1)).real / dq
qq = (np.arange(nq) - nq // 2) * dq
fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
ax.plot(qq, K, color="C1"); ax.axhline(0, color="k", lw=0.5)
ax.set_xlim(-4, 4); ax.set_xlabel("momentum transfer q"); ax.set_ylabel("K(q)")
ax.set_title("Jump kernel at x = 1 over t = T$_0$/50 (signed; coherence window L = 6 for display)", fontsize=9)
fig.savefig(os.path.join(OUT, "kernel.png"), dpi=130)
