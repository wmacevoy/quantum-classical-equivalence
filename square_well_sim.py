#!/usr/bin/env python3
"""
The quantum tunneling correction Delta Q for a square potential: a well of
half-width A between barriers of height V0 and width B (hbar = M = 1).

For piecewise-constant V the classical force is a delta function at each edge
(hard reflection or refraction), and away from the edges

    Delta Q-hat(x, p-hat) = V(x + p-hat/2) - V(x - p-hat/2),

nonzero only once the segment [x - p-hat/2, x + p-hat/2] reaches an edge.  For a
single step of height s at x_j its phase-space kernel is (x != x_j)

    Delta Q(x, q) = -(s / pi hbar) cos(2 (x - x_j) q / hbar) / q,

real, odd in q, with zero zeroth and first moments: it moves signed weight up
and down in momentum without changing probability or the mean force.  Near a
wall it puts weight at momenta above the barrier, which then flies out along
classical paths.

The kernel figure shows Delta Q(x, q) (closed form above), max_q |Delta Q|, which
does not decay away from the barriers, and the same after smoothing over a
momentum resolution dp (Delta Q-hat times exp(-p-hat^2 dp^2 / 2 hbar^2)), which
does.  It also reports the jump rate gamma(x) = int |Delta Q(x, q)| dq for |q| <
q_max, which grows like log q_max.

Three evolutions of the same initial Gaussian (all below the barrier):
  classical flow     sampled classical trajectories (no Delta Q): trapped;
  classical + Delta Q  free flight (FFT in x) and the exact force multiplier
                     exp(i dt [V(x + p-hat/2) - V(x - p-hat/2)] / hbar) in p-hat,
                     i.e. the classical force plus Delta Q, on a phase-space grid;
  exact quantum      split-step Schrodinger on a finer, larger grid.
The grid has p-hat spacing 2 dx so the branch points x +/- p-hat/2 lie on the x
grid.

Outputs (in square_well_sim_out/): square_kernel.png, square_well.png, summary.txt
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "square_well_sim_out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- parameters
HBAR = 1.0
V0, A, B = 8.0, 3.0, 0.5          # barrier height, well half-width, barrier width
EDGES = [(-A - B, +V0), (-A, -V0), (A, +V0), (A + B, -V0)]   # (x_j, step height s_j)
P0, SIGMA = 2.0, 1.0              # initial Gaussian: mean momentum, width parameter
L, N = 100.0, 2048                # phase-space box and grid (N x N)
DT, T_END, NSAMP = 0.005, 10.0, 41
NPART = 400_000


def V(z):
    z = np.abs(z)
    return np.where((z > A) & (z < A + B), V0, 0.0)


def outside(xs, rho, d):
    return rho[np.abs(xs) > A + B].sum() * d


x = -L / 2 + L * np.arange(N) / N
dx = x[1] - x[0]
dp = np.pi * HBAR / (N * dx)
p = (np.arange(N) - N // 2) * dp
kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
phat = 2 * np.pi * HBAR * np.fft.fftfreq(N, d=dp)       # spacing 2 dx
X, PH = np.meshgrid(x, phat, indexing="ij")
KICK = np.exp(1j * DT * (V(X + PH / 2) - V(X - PH / 2)) / HBAR)
DRIFT = np.exp(-1j * np.outer(kx, p) * DT)
HALF = np.exp(-1j * np.outer(kx, p) * DT / 2)
del X, PH

ts = np.linspace(0, T_END, NSAMP)
steps = np.round(ts / DT).astype(int)
ts = steps * DT

# ---------------------------------------------------------------- classical + Delta Q (grid)
Xg, Pg = np.meshgrid(x, p, indexing="ij")
w = np.exp(-Xg**2 / SIGMA**2 - SIGMA**2 * (Pg - P0)**2 / HBAR**2) / (np.pi * HBAR)
del Xg, Pg
born_j = [w.sum(axis=1) * dp]
w = np.fft.ifft(np.fft.fft(w, axis=0) * HALF, axis=0)
for s in range(1, steps[-1] + 1):
    w = np.fft.ifft(np.fft.fft(w, axis=1) * KICK, axis=1)
    if s in steps:
        wt = np.fft.ifft(np.fft.fft(w, axis=0) * HALF, axis=0).real
        born_j.append(wt.sum(axis=1) * dp)
        if s == steps[NSAMP // 2]:
            w_mid = wt.copy()
    w = np.fft.ifft(np.fft.fft(w, axis=0) * DRIFT, axis=0)
born_j = np.array(born_j)
print("classical + Delta Q: done", flush=True)

# ---------------------------------------------------------------- exact quantum
NQ, LQ = 8192, 4 * L
xq = -LQ / 2 + LQ * np.arange(NQ) / NQ
dxq = xq[1] - xq[0]
kq = 2 * np.pi * np.fft.fftfreq(NQ, d=dxq)
psi = (np.pi * SIGMA**2) ** -0.25 * np.exp(-xq**2 / (2 * SIGMA**2) + 1j * P0 * xq / HBAR)
dtq = DT / 4
TQ = np.exp(-1j * HBAR * kq**2 / 2 * dtq)
VQ = np.exp(-1j * V(xq) * dtq / 2 / HBAR)
born_q = [np.abs(psi) ** 2]
for s in range(1, 4 * steps[-1] + 1):
    psi = VQ * np.fft.ifft(TQ * np.fft.fft(VQ * psi))
    if s % 4 == 0 and s // 4 in steps:
        born_q.append(np.abs(psi) ** 2)
born_q = np.array(born_q)
print("exact quantum: done", flush=True)

# ---------------------------------------------------------------- classical flow
rng = np.random.default_rng(1)
xp = rng.normal(0, SIGMA / np.sqrt(2), NPART)
pp = rng.normal(P0, HBAR / (SIGMA * np.sqrt(2)), NPART)
above = np.mean(pp**2 / 2 + V(xp) > V0)
out_c = [np.mean(np.abs(xp) > A + B)]
for s in range(1, steps[-1] + 1):
    xn = xp + DT * pp
    ke = pp**2 / 2 - (V(xn) - V(xp))
    refl = ke < 0
    pp = np.where(refl, -pp, np.sign(pp) * np.sqrt(2 * np.maximum(ke, 0)))
    xp = np.where(refl, xp, xn)
    if s in steps:
        out_c.append(np.mean(np.abs(xp) > A + B))
out_c = np.array(out_c)
print("classical flow: done", flush=True)

out_j = np.array([outside(x, b, dx) for b in born_j])
out_q = np.array([outside(xq, b, dxq) for b in born_q])
tv = []
for bj, bq in zip(born_j, born_q):
    tv.append(0.5 * np.abs(np.interp(x, xq, bq) - bj).sum() * dx)
lines = ["Square potential: well |x| < A between barriers V0 of width B (hbar = M = 1)",
         f"V0 = {V0}, A = {A}, B = {B}; Gaussian at x = 0, p0 = {P0}, width parameter {SIGMA}",
         f"grid {N} x {N}, box {L:g}, dx = {dx:.4f}, dp = {dp:.4f}, dt = {DT}",
         f"classical: {NPART} particles; fraction above the barrier = {above:.2e}",
         "",
         "     t    P(outside): classical  classical+dQ   quantum    Born TV(classical+dQ vs quantum)"]
for i in range(0, NSAMP, 4):
    lines.append(f"  {ts[i]:5.2f}   {out_c[i]:20.5f} {out_j[i]:13.5f} {out_q[i]:10.5f}   {tv[i]:.2e}")
summary = "\n".join(lines)
print(summary)
open(os.path.join(OUT, "summary.txt"), "w").write(summary + "\n")

# ---------------------------------------------------------------- kernel
def dQ(xv, q):
    """Delta Q(x, q) in closed form, a sum over the edges (x off the edges)."""
    return sum(-(sj / (np.pi * HBAR)) * np.cos(2 * (xv - xj) * q / HBAR) / q for xj, sj in EDGES)


def dQ_smoothed(xv, dp_res):
    """Delta Q(x, q) smoothed over momentum resolution dp_res, via its p-hat form."""
    nh, dh = 1 << 14, 0.005
    ph = (np.arange(nh) - nh // 2) * dh
    h = (V(xv + ph / 2) - V(xv - ph / 2)) * np.exp(-(ph * dp_res / HBAR) ** 2 / 2)
    k = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(1j / HBAR * h))) * nh * dh / (2 * np.pi * HBAR)
    return k.real


xk = np.linspace(-10, 10, 1601)
xk = xk[np.min([np.abs(xk - xj) for xj, _ in EDGES], axis=0) > 1e-9]
qk = np.linspace(-6, 6, 1201)
qk = qk[np.abs(qk) > 1e-9]
qfine = np.linspace(1e-6, 64, 400_001)
dqf = qfine[1] - qfine[0]
maxraw = np.array([np.abs(dQ(xv, qfine)).max() for xv in xk])
DP_RES = (0.7, 2.0)
maxsm = {d: np.array([np.abs(dQ_smoothed(xv, d)).max() for xv in xk]) for d in DP_RES}
rate_lines = ["", "jump rate gamma(x) = int_{|q| < q_max} |Delta Q(x, q)| dq",
              "  q_max    x = 1     2   2.8     4     6"]
for qmax in (8, 16, 64, 256):
    qq = np.linspace(1e-6, qmax, 2_000_001)
    g = [2 * np.abs(dQ(xv, qq)).sum() * (qq[1] - qq[0]) for xv in (1, 2, 2.8, 4, 6)]
    rate_lines.append(f"  {qmax:5d}  " + " ".join(f"{v:5.1f}" for v in g))
rate = "\n".join(rate_lines)
print(rate)
open(os.path.join(OUT, "summary.txt"), "a").write(rate + "\n")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), constrained_layout=True)
ax = axes[0]
xs = np.linspace(-6, 6, 1201)
ph = np.linspace(-12, 12, 1201)
XS, PHS = np.meshgrid(xs, ph)
ax.pcolormesh(xs, ph, V(XS + PHS / 2) - V(XS - PHS / 2), cmap="RdBu_r", vmin=-V0, vmax=V0,
              shading="auto", rasterized=True)
for xj, _ in EDGES:
    ax.axvline(xj, color="k", lw=0.6)
ax.set_xlabel("x"); ax.set_ylabel(r"$\hat{p}$ (branch separation $z_+ - z_-$)")
ax.set_title(r"(a) $\Delta\hat{Q}(x,\hat{p})$", fontsize=10)

ax = axes[1]
XK, QK = np.meshgrid(xk, qk, indexing="ij")
ax.pcolormesh(xk, qk, dQ(XK, QK).T, cmap="RdBu_r", vmin=-3, vmax=3, shading="auto", rasterized=True)
for xj, _ in EDGES:
    ax.axvline(xj, color="k", lw=0.6)
ax.set_xlabel("x"); ax.set_ylabel("momentum jump q")
ax.set_title(r"(b) $\Delta Q(x,q)$", fontsize=10)

ax = axes[2]
ax.plot(xk, maxraw, "k", label="unsmoothed")
for d, c in zip(DP_RES, ("C0", "C1")):
    ax.plot(xk, maxsm[d], c, label=rf"smoothed over $\Delta p$ = {d:g}")
for xj, _ in EDGES:
    ax.axvline(xj, color="k", lw=0.6)
ax.set_xlabel("x"); ax.legend(fontsize=8)
ax.set_title(r"(c) $\max_q|\Delta Q(x,q)|$", fontsize=10)
fig.savefig(os.path.join(OUT, "square_kernel.png"), dpi=130)

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), constrained_layout=True)
sel = np.abs(x) < 12
psel = np.abs(p) < 7
ax = axes[0]
lim = np.max(np.abs(w_mid)) * 0.6
ax.pcolormesh(x[sel], p[psel], w_mid[np.ix_(sel, psel)].T, cmap="RdBu_r", vmin=-lim, vmax=lim,
              shading="auto", rasterized=True)
for xj, _ in EDGES:
    ax.axvline(xj, color="k", lw=0.6)
ax.axhline(np.sqrt(2 * V0), color="k", ls=":", lw=0.8)
ax.axhline(-np.sqrt(2 * V0), color="k", ls=":", lw=0.8)
ax.set_xlabel("x"); ax.set_ylabel("p")
ax.set_title(f"(a) w(x, p) at t = {ts[NSAMP // 2]:.0f}, classical + $\\Delta Q$", fontsize=10)

ax = axes[1]
ax.plot(ts, out_q, "k", lw=2.5, label="exact quantum")
ax.plot(ts, out_c, "C0--", label=r"classical flow (no $\Delta Q$)")
ax.plot(ts, out_j, "C1", lw=1, label=r"classical + $\Delta Q$")
ax.set_xlabel("t"); ax.set_ylabel("probability outside the well")
ax.set_title("(b) escape from the well", fontsize=10)
ax.legend(fontsize=8)
fig.savefig(os.path.join(OUT, "square_well.png"), dpi=130)
