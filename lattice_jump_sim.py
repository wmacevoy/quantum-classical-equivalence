#!/usr/bin/env python3
"""
Discrete momentum jumps in a lattice, V = -V0 cos(k x): quantum dynamics as
classical free flight plus a force that acts by finite jumps of +/- hbar k / 2.

For one Fourier mode the force term of the w-hat equation is exact without any
expansion:

    (i/hbar)[V(x + p-hat/2) - V(x - p-hat/2)] = (2 i V0 / hbar) sin(kx) sin(k p-hat / 2),

and in phase space this is a centred difference in p with step hbar k:

    w_t |force = (V0 sin kx / hbar) [ w(x, p + hbar k/2) - w(x, p - hbar k/2) ].

Classically (hbar -> 0) the difference becomes hbar k w_p and the term becomes
V'(x) w_p, the Liouville force term.  Read as particles: at rate
r = V0 sin(kx)/hbar, weight at p sends a + copy to p - hbar k/2 and a - copy to
p + hbar k/2.  The signed pair carries momentum -r hbar k = -V'(x) per unit time
on average (the classical force); its higher moments are the quantum correction.

The scheme below is a Strang split: exact free flight (FFT in x) and the jump
equation integrated by RK4 using np.roll in p (hbar k / 2 is a whole number of
momentum cells).  The "classical flow" baseline is the same initial w carried
by sampled classical particle trajectories (leapfrog), binned cloud-in-cell;
a grid Liouville solver is not used because classical filaments become finer
than any fixed momentum grid.

Test (hbar = M = 1, lattice spacing 1, k = 2 pi, V0 = 20): a Gaussian of width
1.5 sites at rest, centred on a well, in a periodic box of 64 sites.  Classically
most particles start below the barrier and stay trapped in their wells; quantum
mechanically the packet spreads through the lattice by tunnelling.  The
reference is exact evolution in the plane-wave eigenbasis of the same box.

Numerical notes.  The (x, p) grid with x periodic in L represents each branch
psi+- on a ring of length 2L, while the quantum reference lives on a ring of
length L; the two agree only until the spreading packet's tails reach the box
edge, hence the 64-site box.  Snapshots are compared at the actual step times.

Outputs (in lattice_jump_sim_out/): born.png, spread.png, summary.txt
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lattice_jump_sim_out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- parameters
HBAR = 1.0
K = 2 * np.pi            # lattice wavenumber (spacing 1)
V0 = 20.0
L = 64.0                 # periodic box, 64 sites
NX = 2048
M_SHIFT = 64             # hbar k / 2 = M_SHIFT momentum cells
NP = 2048
SIGMA = 1.5
T_END = 3.0
T_OSC = 2 * np.pi / (K * np.sqrt(V0))   # small-oscillation period in a well
DT = T_OSC / 200
NSAMP = 31

x = -L / 2 + L * np.arange(NX) / NX
dx = x[1] - x[0]
dp = HBAR * K / 2 / M_SHIFT
p = (np.arange(NP) - NP // 2) * dp
kx = 2 * np.pi * np.fft.fftfreq(NX, d=dx)
DRIFT = np.exp(-1j * np.outer(kx, p) * DT / 2)      # half drift: w(x - p dt/2, p)
RATE = (V0 * np.sin(K * x) / HBAR)[:, None]          # jump rate r(x)
NPART = 400_000


def jump_rhs(w):
    """(V0 sin kx / hbar) [w(p + hbar k/2) - w(p - hbar k/2)]."""
    return RATE * (np.roll(w, -M_SHIFT, axis=1) - np.roll(w, M_SHIFT, axis=1))


def jumps(w, h):
    k1 = jump_rhs(w)
    k2 = jump_rhs(w + h / 2 * k1)
    k3 = jump_rhs(w + h / 2 * k2)
    k4 = jump_rhs(w + h * k3)
    return w + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def drift(w):
    return np.fft.ifft(np.fft.fft(w, axis=0) * DRIFT, axis=0).real


def step(w):
    return drift(jumps(drift(w), DT))


def force(xp):
    return -V0 * K * np.sin(K * xp)


def cic(xp):
    """Cloud-in-cell density of particles on the periodic x grid."""
    u = (xp - x[0]) / dx
    i = np.floor(u).astype(int)
    f = u - i
    rho = np.bincount(i % NX, 1 - f, NX) + np.bincount((i + 1) % NX, f, NX)
    return rho / (xp.size * dx)


# ---------------------------------------------------------------- exact quantum reference
NB = 1280
n = np.arange(-NB // 2, NB // 2)
pn = 2 * np.pi * HBAR * n / L
G = int(round(K * L / (2 * np.pi)))                   # V couples n to n +/- G
H = np.diag(pn**2 / 2)
H[np.arange(NB - G), np.arange(G, NB)] = -V0 / 2
H[np.arange(G, NB), np.arange(NB - G)] = -V0 / 2
E, U = np.linalg.eigh(H)
psi0 = (np.pi * SIGMA**2) ** -0.25 * np.exp(-x**2 / (2 * SIGMA**2))
basis = np.exp(1j * np.outer(x, pn) / HBAR) / np.sqrt(L)     # (NX, NB)
c0 = basis.conj().T @ psi0 * dx
captured = float(np.sum(np.abs(c0) ** 2))
a0 = U.conj().T @ c0


def born_quantum(t):
    psi = basis @ (U @ (a0 * np.exp(-1j * E * t / HBAR)))
    return np.abs(psi) ** 2


# ---------------------------------------------------------------- runs
X, P = np.meshgrid(x, p, indexing="ij")
w0 = np.exp(-X**2 / SIGMA**2 - SIGMA**2 * P**2 / HBAR**2) / (np.pi * HBAR)
ts = np.linspace(0, T_END, NSAMP)
sample_steps = np.round(ts / DT).astype(int)
ts = sample_steps * DT                                # actual sample times
runs = {}
w, snaps, k = w0.copy(), [], 0
for s in range(sample_steps[-1] + 1):
    if s == sample_steps[k]:
        snaps.append(w.sum(axis=1) * dp)
        k += 1
        if k == NSAMP:
            break
    w = step(w)
runs["trajectories + jumps"] = np.array(snaps)
print("trajectories + jumps: done", flush=True)

rng = np.random.default_rng(1)
xp = rng.normal(0, SIGMA / np.sqrt(2), NPART)
pp = rng.normal(0, HBAR / (SIGMA * np.sqrt(2)), NPART)
free = np.mean(pp**2 / 2 - V0 * np.cos(K * xp) > V0)
snaps, k = [], 0
for s in range(sample_steps[-1] + 1):
    if s == sample_steps[k]:
        snaps.append(cic(xp))
        k += 1
        if k == NSAMP:
            break
    pp += DT / 2 * force(xp)
    xp = (xp + DT * pp + L / 2) % L - L / 2
    pp += DT / 2 * force(xp)
runs = {"classical flow": np.array(snaps), **runs}
print("classical flow: done", flush=True)
born_q = np.array([born_quantum(t) for t in ts])
tv = {lab: 0.5 * np.abs(b - born_q).sum(axis=1) * dx for lab, b in runs.items()}
rms = {lab: np.sqrt((b * x**2).sum(axis=1) * dx) for lab, b in list(runs.items()) + [("exact quantum", born_q)]}

lines = ["Lattice V = -V0 cos(kx), quantum force as jumps of +/- hbar k/2 (hbar = M = 1)",
         f"k = 2 pi, V0 = {V0}, box {L:g} sites, grid {NX} x {NP}, dp = hbar k / {2 * M_SHIFT},"
         f" dt = T_osc/200, T_osc = {T_OSC:.4f}",
         f"initial Gaussian width {SIGMA} sites at rest; plane-wave basis captures {captured:.8f}",
         f"classical: {NPART} particles, fraction above the barrier (untrapped) = {free:.4f}",
         "",
         "     t      rms x: quantum  classical  jumps      Born TV: classical  jumps"]
for i in range(0, NSAMP, 3):
    lines.append(f"  {ts[i]:5.2f}   {rms['exact quantum'][i]:12.4f} {rms['classical flow'][i]:10.4f}"
                 f" {rms['trajectories + jumps'][i]:7.4f}   {tv['classical flow'][i]:16.5f}"
                 f"  {tv['trajectories + jumps'][i]:.2e}")
summary = "\n".join(lines)
print(summary)
open(os.path.join(OUT, "summary.txt"), "w").write(summary + "\n")

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
for ax, i in zip(axes, [NSAMP // 6, NSAMP // 2, NSAMP - 1]):
    ax.plot(x, born_q[i], "k", lw=2.5, label="exact quantum")
    ax.plot(x, runs["classical flow"][i], "C0--", lw=1, label="classical flow")
    ax.plot(x, runs["trajectories + jumps"][i], "C1", lw=1, label="trajectories + jumps")
    ax.set_xlim(-12, 12); ax.set_xlabel("x (lattice sites)"); ax.set_title(f"t = {ts[i]:.1f}", fontsize=10)
axes[0].legend(fontsize=8)
fig.savefig(os.path.join(OUT, "born.png"), dpi=130)

fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
ax.plot(ts, rms["exact quantum"], "k", lw=2.5, label="exact quantum")
ax.plot(ts, rms["classical flow"], "C0--", label="classical flow (trapped)")
ax.plot(ts, rms["trajectories + jumps"], "C1", label="trajectories + jumps")
ax.set_xlabel("t"); ax.set_ylabel("rms x (lattice sites)"); ax.legend(fontsize=8)
fig.savefig(os.path.join(OUT, "spread.png"), dpi=130)
