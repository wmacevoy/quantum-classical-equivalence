#!/usr/bin/env python3
"""
Quantum eigenstates under classical Liouville flow: how big is the dropped term?

An energy eigenstate is trivial quantum mechanically: its Wigner function w_n
never changes.  Classically, the same w_n is just an initial phase-space
distribution, and Liouville flow moves it.  For an eigenstate the quantum
equation reads

    p w_x - V' w_p = -(hbar^2/24) V''' w_ppp          (w_t = 0),

so the classical rate of change is exactly the dropped error term:

    w_t(classical) = -(p w_x - V' w_p) = (hbar^2/24) V''' w_ppp .

For the quartic well V = x^4/4 (the slide-6 cloud demo), V''' = 6x and
V^(5) = 0, so the Moyal series terminates: this single term is the ENTIRE
difference between the two dynamics, not just its leading part.

The script measures, for eigenstates n = 0 (most quantum) and n = 12
(semiclassical):
  1. Instantaneous size: ||error term|| relative to the individual Liouville
     flux terms ||p w_x|| and ||V' w_p|| that must cancel, and the drift per
     classical period ||w_t|| T_n / ||w_n||.  Also checks the identity above
     numerically (validates the Wigner functions and derivatives).
  2. Finite-time drift: pushes w_n backward along exact Hamiltonian
     trajectories, w(z, t) = w_n(Phi_{-t} z), and tracks the overlap with the
     stationary quantum answer and the drift of the Born position density.
  3. A sweep over n of the instantaneous ratios.

Outputs (in eigenstate_drift_sim_out/):
  snapshots.png   - classical Liouville evolution of w_0 and w_12 (quantum: frozen)
  drift_vs_t.png  - overlap and Born-density drift vs time (in classical periods)
  ratio_vs_n.png  - relative size of the dropped term vs n
  summary.txt     - numerical table

Units: hbar = M = 1.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.integrate import quad

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eigenstate_drift_sim_out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- parameters
SHOW = [0, 12]             # eigenstates evolved in time
SWEEP = list(range(0, 21)) # eigenstates for the instantaneous-ratio sweep
NG = 2400                  # 1D grid for the eigenproblem
XL = 7.0                   # eigenproblem box [-XL, XL]
NPS = 321                  # phase-space grid per axis (Wigner function, pictures)
NF = 720                   # initial cell-centered grid per axis pushed forward (3 cells per bin)
NBIN = 240                 # position bins for the Born density
PERIODS = 6.0              # classical periods to evolve
NT = 241                   # time samples for the drift curves
SNAP = [0.0, 0.25, 1.0, 6.0]  # snapshot times, in classical periods


def V(x):
    return 0.25 * x**4


def dV(x):
    return x**3


def d3V(x):
    return 6.0 * x


# ---------------------------------------------------------------- eigenstates
xg = np.linspace(-XL, XL, NG)
h = xg[1] - xg[0]
E_all, vecs = eigh_tridiagonal(1.0 / h**2 + V(xg), -0.5 / h**2 * np.ones(NG - 1),
                               select="i", select_range=(0, max(SWEEP + SHOW)))
vecs /= np.sqrt(h)


def psi_spline(n):
    v = vecs[:, n]
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return CubicSpline(xg, v)


def period(E):
    """Classical period of H = p^2/2 + x^4/4 at energy E."""
    xt = (4 * E) ** 0.25
    f = lambda x: 1.0 / np.sqrt(2 * (E - V(x)))
    return 4 * quad(f, 0, xt, limit=200)[0]


# ---------------------------------------------------------------- Wigner
def wigner_parts(n, x, p):
    """w, w_x, w_p, w_ppp of eigenstate n on the grid x by p (real psi).

    w(x,p) = (1/pi) int psi(x+y) psi(x-y) cos(2py) dy, hbar = 1."""
    s = psi_spline(n)
    ds = s.derivative()
    ymax = XL
    y = np.linspace(0, ymax, 1600)
    dy = y[1] - y[0]
    wt = np.full_like(y, dy); wt[0] = wt[-1] = dy / 2
    wt *= 2.0 / np.pi                       # even integrand: int_{-inf}^{inf} = 2 int_0^inf
    X, Y = np.meshgrid(x, y, indexing="ij")
    a, b = X + Y, X - Y
    inside = (np.abs(a) < XL) & (np.abs(b) < XL)
    f = np.where(inside, s(a) * s(b), 0.0) * wt
    fx = np.where(inside, ds(a) * s(b) + s(a) * ds(b), 0.0) * wt
    C = np.cos(2 * np.outer(y, p))
    S = np.sin(2 * np.outer(y, p))
    w = f @ C
    w_x = fx @ C
    w_p = (f * (-2 * y)) @ S
    w_ppp = (f * (2 * y) ** 3) @ S
    return w, w_x, w_p, w_ppp


def ps_grid(n):
    E = E_all[n]
    xm = 1.35 * (4 * E) ** 0.25 + 1.5
    pm = 1.35 * np.sqrt(2 * E) + 1.5
    return np.linspace(-xm, xm, NPS), np.linspace(-pm, pm, NPS)


def instantaneous(n):
    x, p = ps_grid(n)
    w, w_x, w_p, w_ppp = wigner_parts(n, x, p)
    X, P = np.meshgrid(x, p, indexing="ij")
    flux_x = P * w_x                         # p w_x
    flux_p = dV(X) * w_p                     # V' w_p
    err = -(1.0 / 24) * d3V(X) * w_ppp       # -(hbar^2/24) V''' w_ppp
    resid = (flux_x - flux_p) - err          # should vanish for an eigenstate
    nrm = lambda a: np.sqrt(np.sum(a**2))
    T = period(E_all[n])
    return dict(
        n=n, E=E_all[n], T=T,
        err_over_flux=nrm(err) / nrm(flux_x),
        drift_per_period=nrm(err) * T / nrm(w),
        resid_over_err=nrm(resid) / nrm(err),
        neg=-w.min() / w.max(),
    )


# ---------------------------------------------------------------- Liouville flow
def flow(x0, p0, t, dt):
    """Carry points along Hamilton's flow for (signed) time t, velocity Verlet."""
    x, p = x0.copy(), p0.copy()
    steps = int(np.ceil(abs(t) / dt))
    if steps == 0:
        return x, p
    d = t / steps
    for _ in range(steps):
        p -= 0.5 * d * dV(x)
        x += d * p
        p -= 0.5 * d * dV(x)
    return x, p


def cic(pos, wts, edges):
    """Cloud-in-cell deposit of weighted points onto bin centers (linear
    weighting).  Unlike a nearest-bin histogram it does not alias when the
    sample spacing is not commensurate with the bins."""
    nb = edges.size - 1
    d = edges[1] - edges[0]
    u = (pos - (edges[0] + d / 2)) / d
    i = np.floor(u).astype(int)
    f = u - i
    out = np.zeros(nb + 2)
    for k, wk in ((i, wts * (1 - f)), (i + 1, wts * f)):
        ok = (k >= -1) & (k <= nb)
        out += np.bincount(k[ok] + 1, weights=wk[ok], minlength=nb + 2)
    return out[1:-1]


def evolve(n):
    """Classical Liouville evolution of w_n.

    Measurements push a fine initial grid FORWARD with fixed weights w_n dA
    (Liouville carries weight along trajectories), so int w is conserved
    exactly and w_n is only ever evaluated where it is smooth.  Evaluating
    w(z, t) = w_n(Phi_{-t} z) on a fixed grid instead aliases once the
    filaments become finer than the grid; that is used only for the pictures.
    """
    x, p = ps_grid(n)
    w0 = wigner_parts(n, x, p)[0]
    interp = RegularGridInterpolator((x, p), w0, method="cubic",
                                     bounds_error=False, fill_value=0.0)
    xf = x[0] + (np.arange(NF) + 0.5) * (x[-1] - x[0]) / NF
    pf = p[0] + (np.arange(NF) + 0.5) * (p[-1] - p[0]) / NF
    wf = wigner_parts(n, xf, pf)[0]
    Xf, Pf = np.meshgrid(xf, pf, indexing="ij")
    q = (wf * (xf[1] - xf[0]) * (pf[1] - pf[0])).ravel()   # signed weights
    T = period(E_all[n])
    dt = T / 400
    ts = np.linspace(0, PERIODS * T, NT)

    edges = np.linspace(x[0], x[-1], NBIN + 1)
    xc = 0.5 * (edges[1:] + edges[:-1])
    born = lambda xs: cic(xs, q, edges)
    xq = np.linspace(x[0], x[-1], 40 * NBIN + 1)
    rho_q = cic(xq, psi_spline(n)(xq) ** 2 * (xq[1] - xq[0]), edges)   # exact |psi_n|^2
    # coarse graining: Gaussian of std half the node spacing, x_t / (n + 1)
    sig = (4 * E_all[n]) ** 0.25 / (n + 1)
    G = np.exp(-0.5 * ((xc[:, None] - xc[None, :]) / sig) ** 2)
    G /= G.sum(axis=0, keepdims=True)        # columns sum to 1: preserves total weight
    tvd = lambda a, b: 0.5 * np.sum(np.abs(a - b))
    ovl_den = np.sum(q * interp(np.column_stack([Xf.ravel(), Pf.ravel()])))

    overlap, tv, tvc = [], [], []
    xs, ps, tprev = Xf.ravel(), Pf.ravel(), 0.0
    for t in ts:
        xs, ps = flow(xs, ps, t - tprev, dt)  # autonomous flow: compose increments
        tprev = t
        # int w(t) w_n dz = int w_n(z0) w_n(Phi_t z0) dz0   (area preserving)
        overlap.append(np.sum(q * interp(np.column_stack([xs, ps]))) / ovl_den)
        rho = born(xs)
        tv.append(tvd(rho, rho_q))
        tvc.append(tvd(G @ rho, G @ rho_q))

    X, P = np.meshgrid(x, p, indexing="ij")
    snaps = {}
    for s in SNAP:
        xb, pb = flow(X.ravel(), P.ravel(), -s * T, dt)
        snaps[s] = interp(np.column_stack([xb, pb])).reshape(X.shape)
    return dict(n=n, T=T, x=x, p=p, ts=ts / T, overlap=np.array(overlap),
                tv=np.array(tv), tvc=np.array(tvc), snaps=snaps, w0=w0, sig=sig)


# ---------------------------------------------------------------- run
inst = {n: instantaneous(n) for n in sorted(set(SWEEP) | set(SHOW))}
runs = {n: evolve(n) for n in SHOW}

lines = ["Quantum eigenstates of V = x^4/4 under classical Liouville flow (hbar = M = 1)",
         "Dropped term: -(hbar^2/24) V''' w_ppp = -(1/4) x w_ppp  (exact: Moyal series ends here)",
         "",
         " n        E_n       T_n   |err|/|p w_x|   drift/period   identity resid   |min w|/max w",
         "-" * 86]
for n in sorted(inst):
    r = inst[n]
    lines.append(f"{n:2d} {r['E']:10.4f} {r['T']:9.4f} {r['err_over_flux']:15.4f} "
                 f"{r['drift_per_period']:14.4f} {r['resid_over_err']:16.2e} {r['neg']:13.4f}")
lines += ["",
          "drift/period = ||w_t(classical)|| * T_n / ||w_n||   (quantum: 0)",
          "identity resid = ||(p w_x - V' w_p) - err|| / ||err||  (numerical check; should be ~0)",
          "",
          "Finite-time classical evolution (quantum answer: overlap 1, Born drift 0):"]
for n in SHOW:
    r = runs[n]
    lines.append(f"  n = {n}  (T = {r['T']:.4f}; coarse-graining std = {r['sig']:.3f}):")
    for k in [0, (NT - 1) // 12, (NT - 1) // 6, (NT - 1) // 2, NT - 1]:
        lines.append(f"    t = {r['ts'][k]:5.2f} T   overlap = {r['overlap'][k]:7.4f}   "
                     f"Born TV = {r['tv'][k]:6.4f}   coarse Born TV = {r['tvc'][k]:6.4f}")
summary = "\n".join(lines)
print(summary)
with open(os.path.join(OUT, "summary.txt"), "w") as fh:
    fh.write(summary + "\n")

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(len(SHOW), len(SNAP), figsize=(3.2 * len(SNAP), 3.2 * len(SHOW)),
                         constrained_layout=True)
for i, n in enumerate(SHOW):
    r = runs[n]
    vmax = np.abs(r["w0"]).max()
    for j, s in enumerate(SNAP):
        ax = axes[i, j]
        ax.imshow(r["snaps"][s].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  extent=[r["x"][0], r["x"][-1], r["p"][0], r["p"][-1]], aspect="auto")
        E = E_all[n]
        xs = np.linspace(-(4 * E) ** 0.25, (4 * E) ** 0.25, 400)
        ps = np.sqrt(np.maximum(2 * (E - V(xs)), 0))
        ax.plot(xs, ps, "k:", lw=0.6); ax.plot(xs, -ps, "k:", lw=0.6)
        ax.set_title(f"n = {n}, t = {s:g} T" + ("  (= quantum, all t)" if s == 0 else ""),
                     fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("p")
fig.suptitle("Classical Liouville flow of quantum eigenstate Wigner functions, V = x$^4$/4\n"
             "(dotted: classical orbit at E$_n$; quantum evolution leaves the t = 0 panel unchanged)",
             fontsize=10)
fig.savefig(os.path.join(OUT, "snapshots.png"), dpi=130)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
for n in SHOW:
    r = runs[n]
    a1.plot(r["ts"], r["overlap"], label=f"n = {n}")
    c = a2.plot(r["ts"], r["tv"], label=f"n = {n}")[0].get_color()
    a2.plot(r["ts"], r["tvc"], "--", color=c, label=f"n = {n}, coarse-grained")
a1.axhline(1, color="k", lw=0.6, ls=":"); a1.set_ylabel(r"$\int w(t)\,w_n / \int w_n^2$")
a2.set_ylabel("Born position density TV distance")
for a in (a1, a2):
    a.set_xlabel("t / T$_n$ (classical period)"); a.legend()
a1.set_title("Overlap with the (stationary) quantum answer")
a2.set_title("Drift of the observable position density")
fig.savefig(os.path.join(OUT, "drift_vs_t.png"), dpi=130)

fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
ns = sorted(inst)
ax.plot(ns, [inst[n]["err_over_flux"] for n in ns], "o-", label=r"$\|$err$\|$ / $\|p\,w_x\|$")
ax.plot(ns, [inst[n]["drift_per_period"] for n in ns], "s-", label="drift per period")
ax.set_xlabel("eigenstate n"); ax.set_yscale("log"); ax.legend()
ax.set_title("Relative size of the dropped term, V = x$^4$/4")
fig.savefig(os.path.join(OUT, "ratio_vs_n.png"), dpi=130)
