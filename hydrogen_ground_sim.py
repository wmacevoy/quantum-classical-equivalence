#!/usr/bin/env python3
"""
Hydrogen 1s under classical Liouville flow.

Quantum mechanically the 1s state is stationary.  Classically its Wigner
function is an initial distribution in 6D phase space, which Liouville flow
in the Coulomb potential V = -1/r moves.  Unlike the quartic well, the Moyal
series for 1/r never terminates (every odd derivative of V contributes, and
all are singular at r = 0), so the classical drift here is the full sum of
the dropped terms, not just the hbar^2 one.

Reduction (exact, no radial approximation).  The 1s Wigner function is
rotation invariant, so it depends only on r = |r|, p_r = r.p/r and
L = |r x p|.  The phase-space measure is
    d^3r d^3p = 8 pi^2 L dr dp_r dL ,
L is conserved, and each (r, p_r, L) moves in the radial Kepler problem with
V_eff = L^2/(2 r^2) - 1/r.  Orbits are propagated analytically (Kepler's
equation; elliptic and hyperbolic), so there is no time-stepping error.

Note that the classical ensemble is NOT the L = 0 plunge orbits: the Weyl
symbol of L^2 is (r x p)^2 - 3 hbar^2/2, so the 1s Wigner ensemble has
<L^2> = 3/2, and its orbits are Kepler ellipses (and hyperbolas) that mostly
miss the nucleus.

The 1s Wigner function (atomic units, psi = exp(-r)/sqrt(pi)) via prolate
spheroidal coordinates with foci at +-r, xi = 1 + u/r:
    W(r, p_r, L) = (2/pi^3) e^{-2r} int_0^inf du e^{-2u} int_{-1}^{1} deta
                   [(r+u)^2 - r^2 eta^2] cos(2 p_r eta (r+u))
                   J0(2 (L/r) sqrt(u (2r+u) (1-eta^2))) .
It is set to zero for |p| > P_CUT, where the exact 1s momentum density has
~6e-5 of its weight and the quadrature is unreliable.

Outputs (in hydrogen_ground_sim_out/):
  snapshots.png   - L-integrated density in (r, p_r): classical flow vs time
  marginals.png   - radial and momentum densities, classical vs quantum
  drift_vs_t.png  - TV distances and remaining weight vs time
  summary.txt     - numerical table

Units: atomic units (hbar = m_e = e = 1).  T = 2 pi is the Kepler period at
the 1s energy E = -1/2.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import j0
from scipy.interpolate import RegularGridInterpolator
from numpy.polynomial.legendre import leggauss

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hydrogen_ground_sim_out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- parameters
R_MAX, P_MAX, L_MAX = 10.0, 5.0, 8.0   # Wigner grid window
NR, NP, NL = 100, 100, 80              # Wigner grid cells (cell centered)
P_CUT = 8.0                            # zero W above this |p|
NU, NETA, U_MAX = 120, 80, 10.0        # quadrature (converged: 200 x 160 agrees)
REFINE = 1.5                           # second lattice for the noise floor
R_BOX = 20.0                           # bins for densities extend to here
NBIN_R, NBIN_P = 200, 160              # density bins (r in [0, R_BOX], p in [0, 8])
T = 2 * np.pi                          # Kepler period at E = -1/2
NT = 50                                # log-spaced times in [0.01, 100] T, plus 0
SNAP = [0.0, 0.25, 1.0, 10.0]          # snapshot times, in T

clock = time.time()
def log(msg):
    print(f"[{time.time() - clock:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- 1s Wigner function
def wigner_1s(r, pr, L):
    gu, wu = leggauss(NU)
    u = (gu + 1) * U_MAX / 2
    wu = wu * U_MAX / 2
    ge, we = leggauss(NETA)
    U, Et = np.meshgrid(u, ge, indexing="ij")
    base = (np.outer(wu, we) * np.exp(-2 * U)).ravel()
    U, Et = U.ravel(), Et.ravel()
    out = np.zeros(r.size)
    ok = np.sqrt(pr**2 + (L / r) ** 2) <= P_CUT
    idx = np.flatnonzero(ok)
    for a in range(0, idx.size, 2000):
        k = idx[a:a + 2000]
        rr, pp, LL = r[k, None], pr[k, None], L[k, None]
        g = (rr + U) ** 2 - rr**2 * Et**2
        c = np.cos(2 * pp * Et * (rr + U))
        jj = j0(2 * (LL / rr) * np.sqrt(U * (2 * rr + U) * (1 - Et**2)))
        out[k] = (2 / np.pi**3) * np.exp(-2 * rr[:, 0]) * np.sum(base * g * c * jj, axis=1)
    return out


def centers(lo, hi, n):
    return lo + (np.arange(n) + 0.5) * (hi - lo) / n


rc = centers(0, R_MAX, NR)
pc_ = centers(0, P_MAX, NP // 2)                 # W is even in p_r: compute p_r > 0
Lc = centers(0, L_MAX, NL)
R3, P3, L3 = [a.ravel() for a in np.meshgrid(rc, pc_, Lc, indexing="ij")]
w_half = wigner_1s(R3, P3, L3).reshape(NR, NP // 2, NL)
w_grid = np.concatenate([w_half[:, ::-1, :], w_half], axis=1)   # p_r from -P_MAX to P_MAX
pc = np.concatenate([-pc_[::-1], pc_])
log("1s Wigner function on the grid done")


def kepler(r, pr, L, t):
    """Propagate radial Kepler motion (V_eff = L^2/2r^2 - 1/r) for time t."""
    E = 0.5 * pr**2 + 0.5 * L**2 / r**2 - 1.0 / r
    r_t, pr_t = np.empty_like(r), np.empty_like(r)
    b = E < 0
    # elliptic
    a = -0.5 / E[b]
    e = np.sqrt(np.maximum(0.0, 1 + 2 * E[b] * L[b] ** 2))
    ec, es = 1 - r[b] / a, pr[b] * r[b] / np.sqrt(a)
    Ea = np.arctan2(es, ec)
    M = Ea - es + a**-1.5 * t
    M = np.mod(M + np.pi, 2 * np.pi) - np.pi
    Ea = M + 0.85 * e * np.sign(np.sin(M))
    for _ in range(60):
        Ea -= np.clip((Ea - e * np.sin(Ea) - M) / (1 - e * np.cos(Ea)), -1, 1)
    r_t[b] = a * (1 - e * np.cos(Ea))
    pr_t[b] = np.sqrt(a) * e * np.sin(Ea) / r_t[b]
    # hyperbolic
    h = ~b
    a = 0.5 / E[h]
    e = np.sqrt(1 + 2 * E[h] * L[h] ** 2)
    F = np.arcsinh(pr[h] * r[h] / (np.sqrt(a) * e))
    M = e * np.sinh(F) - F + a**-1.5 * t
    F = np.arcsinh(M / e)
    for _ in range(60):
        F -= np.clip((e * np.sinh(F) - F - M) / (e * np.cosh(F) - 1), -1, 1)
    r_t[h] = a * (e * np.cosh(F) - 1)
    pr_t[h] = np.sqrt(a) * e * np.sinh(F) / r_t[h]
    return r_t, pr_t


def cic(pos, wts, edges):
    """Cloud-in-cell deposit of weighted points onto bin centers."""
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


def cic2(x, y, wts, ex, ey):
    """2D cloud-in-cell deposit onto bin centers (bilinear weighting)."""
    nx, ny = ex.size - 1, ey.size - 1
    dx, dy = ex[1] - ex[0], ey[1] - ey[0]
    u = (x - (ex[0] + dx / 2)) / dx
    v = (y - (ey[0] + dy / 2)) / dy
    i, j = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - i, v - j
    out = np.zeros((nx + 2, ny + 2))
    for di, wu in ((0, 1 - fu), (1, fu)):
        for dj, wv in ((0, 1 - fv), (1, fv)):
            ii, jj = i + di, j + dj
            ok = (ii >= -1) & (ii <= nx) & (jj >= -1) & (jj <= ny)
            np.add.at(out, (ii[ok] + 1, jj[ok] + 1), (wts * wu * wv)[ok])
    return out[1:-1, 1:-1]


edges_r = np.linspace(0, R_BOX, NBIN_R + 1)
edges_p = np.linspace(0, 8, NBIN_P + 1)
rb = 0.5 * (edges_r[1:] + edges_r[:-1])
pb = 0.5 * (edges_p[1:] + edges_p[:-1])
# exact quantum marginals of 1s (stationary), as CIC deposits of a fine sampling
xs_ = np.linspace(0, R_BOX, 40 * NBIN_R + 1)
q_r = cic(xs_, 4 * xs_**2 * np.exp(-2 * xs_) * (xs_[1] - xs_[0]), edges_r)
ps_ = np.linspace(0, 8, 40 * NBIN_P + 1)
q_p = cic(ps_, 32 * ps_**2 / (np.pi * (1 + ps_**2) ** 4) * (ps_[1] - ps_[0]), edges_p)

ts = np.concatenate([[0.0], np.geomspace(0.01, 100, NT)]) * T
interp = RegularGridInterpolator((rc, pc, Lc), w_grid, bounds_error=False, fill_value=None)


def lattice(scale):
    """Cell-centered lattice of initial points and their weights 8 pi^2 L w dV."""
    nr, npr, nl = int(NR * scale), int(NP * scale), int(NL * scale)
    r0, p0, l0 = centers(0, R_MAX, nr), centers(-P_MAX, P_MAX, npr), centers(0, L_MAX, nl)
    R, P, L = [a.ravel() for a in np.meshgrid(r0, p0, l0, indexing="ij")]
    if scale == 1:
        w = w_grid.ravel()
    else:
        w = interp(np.column_stack([R, P, L]))
    dV = (R_MAX / nr) * (2 * P_MAX / npr) * (L_MAX / nl)
    return R, P, L, 8 * np.pi**2 * L * w * dV


def run(scale, snaps=False):
    R, P, L, q = lattice(scale)
    E = 0.5 * P**2 + 0.5 * L**2 / R**2 - 1 / R
    res = dict(tv_r=[], tv_p=[], inside=[], margs={}, snaps={}, q=q, E=E, R=R, P=P, L=L)
    for t in ts:
        r_t, pr_t = kepler(R, P, L, t)
        if t == 0:
            res["kepler_check"] = max(np.abs(r_t - R).max(), np.abs(pr_t - P).max())
        p_t = np.sqrt(pr_t**2 + (L / r_t) ** 2)
        cr, cp = cic(r_t, q, edges_r), cic(p_t, q, edges_p)
        res["tv_r"].append(0.5 * np.abs(cr - q_r).sum())
        res["tv_p"].append(0.5 * np.abs(cp - q_p).sum())
        res["inside"].append(q[r_t < R_BOX].sum())
        res["margs"][t / T] = (cr, cp)
    if snaps:
        # bins aligned with the base lattice cells (0.1 x 0.1), so t = 0 is exact
        er = np.linspace(0, 8, 81); ep = np.linspace(-4, 4, 81)
        for s in SNAP:
            r_t, pr_t = kepler(R, P, L, s * T)
            Hh = cic2(r_t, pr_t, q, er, ep)
            res["snaps"][s] = Hh / ((er[1] - er[0]) * (ep[1] - ep[0]))
        res["snap_extent"] = [er[0], er[-1], ep[0], ep[-1]]
    return res


main = run(1, snaps=True)
log("classical run (base lattice) done")
fine = run(REFINE)
log(f"classical run ({REFINE}x lattice) done")

# ---------------------------------------------------------------- checks and moments
q, E, R, P, L = main["q"], main["E"], main["R"], main["P"], main["L"]
# energy conservation of the analytic propagation
r_t, pr_t = kepler(R, P, L, 7.3 * T)
E_t = 0.5 * pr_t**2 + 0.5 * L**2 / r_t**2 - 1 / r_t
ok = np.abs(E) > 1e-3
e_err = np.max(np.abs(E_t[ok] - E[ok]) / np.abs(E[ok]))
w = w_grid.ravel()
neg = q[q < 0].sum()
unb = q[E > 0].sum()

# instantaneous drift rate: w_t = -p_r w_r + V_eff' w_pr,  V_eff' = -L^2/r^3 + 1/r^2
dr, dp = rc[1] - rc[0], pc[1] - pc[0]
Rg, Pg, Lg = np.meshgrid(rc, pc, Lc, indexing="ij")
w_t = -Pg * np.gradient(w_grid, dr, axis=0) + (-Lg**2 / Rg**3 + 1 / Rg**2) * np.gradient(w_grid, dp, axis=1)
meas = 8 * np.pi**2 * Lg
drift_per_period = np.sqrt(np.sum(meas * w_t**2) / np.sum(meas * w_grid**2)) * T

tt = ts / T
lines = ["Hydrogen 1s Wigner function under classical Liouville (Kepler) flow, atomic units",
         f"Grid {NR}x{NP}x{NL} in (r, p_r, L) on [0,{R_MAX}]x[-{P_MAX},{P_MAX}]x[0,{L_MAX}]; |p| <= {P_CUT}",
         "",
         "Checks on the classical ensemble (exact values in parentheses):",
         f"  norm            {q.sum():9.5f}   (1)",
         f"  <H>             {np.sum(q * E):9.5f}   (-0.5)",
         f"  <L^2>           {np.sum(q * L**2):9.5f}   (1.5 = Weyl symbol of L^2 = 0, plus 3/2)",
         f"  <r>             {np.sum(q * R):9.5f}   (1.5)",
         f"  Kepler propagation: t = 0 identity error {main['kepler_check']:.1e}, "
         f"relative energy drift at 7.3 T {e_err:.1e}",
         "",
         f"  negative Wigner weight            {neg:9.5f}",
         f"  weight with E > 0 (unbound)       {unb:9.5f}   <- classically, this part ionizes",
         f"  drift per period ||w_t|| T/||w||  {drift_per_period:9.3f}   (quartic ground state: 2.0)",
         "",
         "Classical vs quantum (quantum: stationary 1s).  TV distances; noise = base vs "
         f"{REFINE}x lattice:",
         "      t/T    radial TV   noise    momentum TV   noise    weight r < 20"]
for k in range(len(ts)):
    if k == 0 or k % 5 == 4 or k == len(ts) - 1:
        nr_ = 0.5 * np.abs(main["margs"][tt[k]][0] - fine["margs"][tt[k]][0]).sum()
        np_ = 0.5 * np.abs(main["margs"][tt[k]][1] - fine["margs"][tt[k]][1]).sum()
        lines.append(f"  {tt[k]:8.2f}   {main['tv_r'][k]:8.4f}   {nr_:7.4f}   {main['tv_p'][k]:10.4f}   "
                     f"{np_:7.4f}   {main['inside'][k]:10.4f}")
summary = "\n".join(lines)
print(summary)
with open(os.path.join(OUT, "summary.txt"), "w") as fh:
    fh.write(summary + "\n")

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(1, len(SNAP), figsize=(3.4 * len(SNAP), 3.4), constrained_layout=True)
vmax = np.abs(main["snaps"][0.0]).max()
for ax, s in zip(axes, SNAP):
    ax.imshow(main["snaps"][s].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
              extent=main["snap_extent"], aspect="auto")
    ax.set_title(f"t = {s:g} T" + ("  (= quantum, all t)" if s == 0 else ""), fontsize=9)
    ax.set_xlabel("r (bohr)"); ax.set_ylabel("p$_r$ (a.u.)")
fig.suptitle("Hydrogen 1s under classical Kepler flow: density in (r, p$_r$), integrated over L",
             fontsize=10)
fig.savefig(os.path.join(OUT, "snapshots.png"), dpi=130)

fig, axes = plt.subplots(2, len(SNAP), figsize=(3.4 * len(SNAP), 5.6), constrained_layout=True)
for j, s in enumerate(SNAP):
    key = min(main["margs"], key=lambda k: abs(k - s))
    cr, cp = main["margs"][key]
    axes[0, j].plot(rb, q_r / np.diff(edges_r), "k", lw=1.6, label="quantum (1s)")
    axes[0, j].plot(rb, cr / np.diff(edges_r), "C1--", lw=1.2, label="classical")
    axes[1, j].plot(pb, q_p / np.diff(edges_p), "k", lw=1.6)
    axes[1, j].plot(pb, cp / np.diff(edges_p), "C1--", lw=1.2)
    axes[0, j].set_title(f"t = {key:.3g} T", fontsize=9)
    axes[0, j].set_xlim(0, 12); axes[1, j].set_xlim(0, 5)
    axes[0, j].set_xlabel("r (bohr)"); axes[1, j].set_xlabel("|p| (a.u.)")
axes[0, 0].set_ylabel("radial density 4$\\pi r^2|\\psi|^2$")
axes[1, 0].set_ylabel("momentum density")
axes[0, 0].legend(fontsize=8)
fig.savefig(os.path.join(OUT, "marginals.png"), dpi=130)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
a1.semilogx(tt[1:], main["tv_r"][1:], "o-", ms=3, label="radial density")
a1.semilogx(tt[1:], main["tv_p"][1:], "s-", ms=3, label="momentum density")
a1.set_ylabel("TV distance, classical vs quantum"); a1.legend()
a2.semilogx(tt[1:], main["inside"][1:], "o-", ms=3, color="C2")
a2.axhline(1 - unb, color="k", ls=":", lw=0.8, label=f"1 - unbound weight = {1 - unb:.3f}")
a2.set_ylabel(f"classical weight with r < {R_BOX:g}"); a2.legend()
for a in (a1, a2):
    a.set_xlabel("t / T  (T = 2$\\pi$, Kepler period at E = -1/2)")
fig.savefig(os.path.join(OUT, "drift_vs_t.png"), dpi=130)
