#!/usr/bin/env python3
"""
A semiclassical state in the quartic well: classical vs quantum evolution.

Initial state: the ground-state Wigner function of V = x^4/4 scaled out by
s = 10 in both x and p,

    w_0(x, p) = W_g(x/s, p/s) / s^2 ,

i.e. the ground-state shape covering s^2 = 100 times the phase-space area of
hbar.  It is smooth on the hbar scale (a positive mixed state), so unlike an
eigenstate it is NOT stationary quantum mechanically: both sides move, and we
compare them directly.

Both evolutions are exact.  Classical: Liouville flow (Hamilton's equations)
of the signed weights w_0 dA.  Quantum: the density matrix in the eigenbasis
of H, rho_mn(t) = rho_mn exp(-i (E_m - E_n) t).  For x^4/4 the Moyal series
ends at the hbar^2 term, so

    w_t(quantum) - w_t(classical) = -(hbar^2/24) V''' w_ppp = -(1/4) x w_ppp

is the whole difference.  The script reports its instantaneous relative size
at t = 0 and then tracks how far the two evolutions separate in time.

Density matrix of the scaled state (closed form, no Wigner inversion):
    rho_0(x, x') = (1/s) psi_g(Xbar/s + s D/2) psi_g(Xbar/s - s D/2),
    Xbar = (x + x')/2,  D = x - x'.
It is projected onto the kept eigenstates; the projected state (whose trace
is reported) is the common initial condition for both sides.

Outputs (in scaled_state_sim_out/):
  snapshots.png   - classical vs quantum Wigner functions at several times
  marginals.png   - position and momentum densities, classical vs quantum
  drift_vs_t.png  - TV distance between classical and quantum marginals vs t
  summary.txt     - numerical table

Units: hbar = M = 1.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.integrate import quad

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scaled_state_sim_out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- parameters
S = 10.0              # scale factor on the ground-state Wigner function
L = 25.0              # quantum box [-L, L]
H_DX = 0.0085         # DVR grid spacing (max momentum pi/H_DX ~ 370)
E_CUT = 4.0e4         # keep eigenstates below this energy
NPS = 401             # phase-space grid per axis (Wigner pictures, classical w_0)
NF = 720              # initial cell-centered grid per axis pushed forward (3 cells per bin)
XW, PW = 24.0, 48.0   # phase-space window
NBIN = 240            # bins for the marginal densities
T_MAX = 100.0         # in units of the classical period at <E>
NT = 50               # log-spaced sample times (plus t = 0)
SNAP = [0.0, 1.0, 10.0, 100.0]  # snapshot times, in periods
LAM_TOL = 1e-7        # drop density-matrix eigenvalues below this (relative)


def V(x):
    return 0.25 * x**4


def dV(x):
    return x**3


def period(E):
    xt = (4 * E) ** 0.25
    return 4 * quad(lambda x: 1.0 / np.sqrt(2 * (E - V(x))), 0, xt, limit=200)[0]


clock = time.time()
def log(msg):
    print(f"[{time.time() - clock:6.1f}s] {msg}", flush=True)


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


# ---------------------------------------------------------------- ground state (unscaled)
xg = np.linspace(-8, 8, 4001)
hg = xg[1] - xg[0]
_, vg = eigh_tridiagonal(1 / hg**2 + V(xg), -0.5 / hg**2 * np.ones(xg.size - 1),
                         select="i", select_range=(0, 0))
vg = np.abs(vg[:, 0]) / np.sqrt(hg)
psi_g = CubicSpline(xg, vg, extrapolate=False)
def psi_g0(u):
    return np.nan_to_num(psi_g(u))


# ---------------------------------------------------------------- eigenbasis (sinc DVR)
x = np.arange(-L, L + H_DX / 2, H_DX)
N = x.size
d = np.arange(N)[:, None] - np.arange(N)[None, :]
with np.errstate(divide="ignore"):
    Tk = np.where(d == 0, np.pi**2 / 3, 2.0 * (-1.0) ** d / np.where(d == 0, 1, d) ** 2)
Hm = Tk / (2 * H_DX**2)
Hm[np.diag_indices(N)] += V(x)
del Tk, d
# number of states below E_CUT (Weyl estimate, padded)
area = 4 * quad(lambda u: np.sqrt(2 * (E_CUT - V(u))), 0, (4 * E_CUT) ** 0.25)[0]
K = int(1.05 * area / (2 * np.pi))
log(f"DVR: N = {N}, diagonalizing for {K} states")
E, Psi = eigh(Hm, subset_by_index=[0, K - 1], driver="evr", overwrite_a=True)
del Hm
keep = E < E_CUT
E, Psi = E[keep], Psi[:, keep] / np.sqrt(H_DX)   # sum |psi|^2 h = 1
K = E.size
log(f"kept {K} states, E_max = {E[-1]:.0f}")

# ---------------------------------------------------------------- initial density matrix
Xb = 0.5 * (x[:, None] + x[None, :])
D = x[:, None] - x[None, :]
rho0 = psi_g0(Xb / S + S * D / 2) * psi_g0(Xb / S - S * D / 2) / S
del Xb, D
rhoE = H_DX**2 * (Psi.T @ rho0 @ Psi)          # rho in the eigenbasis
del rho0
trace = np.trace(rhoE)
lam, A = np.linalg.eigh(rhoE)                  # rho = sum_k lam_k a_k a_k^T
# The scaled Wigner function of a NON-Gaussian state need not be a valid
# density matrix: rho_0 has small negative eigenvalues.  Keep the signed
# mixture (both evolutions are linear) and report the negative weight.
neg_weight = -lam[lam < 0].sum()
neg_min = lam.min()
sel = np.abs(lam) > LAM_TOL * np.abs(lam).max()
lam, A = lam[sel], A[:, sel]
log(f"projected trace = {trace:.6f}; {lam.size} density-matrix eigenvalues kept; "
    f"negative eigenvalues total {-neg_weight:.2e} (most negative {neg_min:.2e})")
lam = lam / lam.sum()                          # renormalize the projected state
Emean = float((E @ A**2) @ lam)
T = period(Emean)
log(f"<E> = {Emean:.1f}, classical period at <E>: T = {T:.4f}")


def quantum_modes(t):
    """Phi[:, k] = psi_k(x, t) (grid values) for the mixture rho = sum lam_k |phi_k><phi_k|."""
    return Psi @ (np.exp(-1j * E * t)[:, None] * A)


# ---------------------------------------------------------------- phase-space grids
xw = np.linspace(-XW, XW, NPS)
pw = np.linspace(-PW, PW, NPS)
edges_x = np.linspace(-XW, XW, NBIN + 1)
edges_p = np.linspace(-PW, PW, NBIN + 1)


def wigner_q(Phi):
    """Wigner function of sum_k lam_k |phi_k><phi_k| (lam signed) on (xw, pw).

    w(x_i, p) = (h/pi) sum_j rho[i+j, i-j] exp(2 i p j h), rho from Phi."""
    idx = np.clip(np.round((xw - x[0]) / H_DX).astype(int), 0, N - 1)
    Jmax = 1600                                   # |x - x'| up to 2*Jmax*h ~ 27
    js = np.arange(-Jmax, Jmax + 1)
    out = np.zeros((NPS, NPS))
    ph = np.exp(2j * np.outer(js * H_DX, pw))     # (2J+1, NPS)
    Wl = Phi * lam[None, :]
    for a, i in enumerate(idx):
        ip, im = i + js, i - js
        ok = (ip >= 0) & (ip < N) & (im >= 0) & (im < N)
        r = np.zeros(js.size, complex)
        r[ok] = np.einsum("jk,jk->j", Wl[ip[ok]], Phi[im[ok]].conj())
        out[a] = (H_DX / np.pi) * np.real(r @ ph)
    return out


def marg_q(Phi):
    """Quantum position and momentum densities, integrated over the bins."""
    rx = (np.abs(Phi) ** 2) @ lam
    px = cic(x, rx * H_DX, edges_x)
    # momentum: DFT of each mode, |phi(p)|^2 with phi(p) = h/sqrt(2 pi) sum phi(x) e^{-ipx}
    M = 8192
    F = np.fft.fftshift(np.fft.fft(Phi, n=M, axis=0), axes=0)
    pk = np.fft.fftshift(np.fft.fftfreq(M, d=H_DX)) * 2 * np.pi
    phase = np.exp(-1j * pk * x[0])[:, None]
    rp = (np.abs(F * phase) ** 2 * H_DX**2 / (2 * np.pi)) @ lam
    pp = cic(pk, rp * (pk[1] - pk[0]), edges_p)
    return px, pp


# ---------------------------------------------------------------- classical side
def flow(x0, p0, t, dt):
    xs, ps = x0.copy(), p0.copy()
    steps = int(np.ceil(abs(t) / dt))
    if steps == 0:
        return xs, ps
    h = t / steps
    for _ in range(steps):
        ps -= 0.5 * h * dV(xs)
        xs += h * ps
        ps -= 0.5 * h * dV(xs)
    return xs, ps


Phi0 = quantum_modes(0.0)
w0 = wigner_q(Phi0)
log("quantum w_0 on the phase-space grid done")
interp0 = RegularGridInterpolator((xw, pw), w0, method="cubic", bounds_error=False, fill_value=0.0)
# check: w_0 should equal the closed form W_g(x/s, p/s)/s^2 (up to the projection)
yv = np.linspace(0, 8, 2001); wy = np.full(yv.size, yv[1] - yv[0]); wy[[0, -1]] /= 2
Xs = xw[:, None] / S
f = psi_g0(Xs + yv[None, :]) * psi_g0(Xs - yv[None, :]) * wy * (2 / np.pi)
w_formula = f @ np.cos(2 * np.outer(yv, pw / S)) / S**2
w0_check = np.abs(w0 - w_formula).max() / np.abs(w_formula).max()
log(f"check: max |w0 - closed form| / max w = {w0_check:.2e}")

# instantaneous size of the dropped term at t = 0 (finite differences of smooth w_0)
dx, dp = xw[1] - xw[0], pw[1] - pw[0]
w_x = np.gradient(w0, dx, axis=0)
w_p = np.gradient(w0, dp, axis=1)
w_ppp = np.gradient(np.gradient(w_p, dp, axis=1), dp, axis=1)
Xw, Pw = np.meshgrid(xw, pw, indexing="ij")
err = -0.25 * Xw * w_ppp
flux = Pw * w_x - dV(Xw) * w_p                  # classical: w_t = -flux
nrm = lambda a: np.sqrt(np.sum(a**2))
ratio_px = nrm(err) / nrm(Pw * w_x)
ratio_wt = nrm(err) / nrm(flux)

xf = -XW + (np.arange(NF) + 0.5) * (2 * XW / NF)
pf = -PW + (np.arange(NF) + 0.5) * (2 * PW / NF)
Xf, Pf = np.meshgrid(xf, pf, indexing="ij")
q = interp0(np.column_stack([Xf.ravel(), Pf.ravel()])) * (xf[1] - xf[0]) * (pf[1] - pf[0])
log(f"classical weights: sum = {q.sum():.6f}")

ts = np.concatenate([[0.0], np.geomspace(0.05, T_MAX, NT)]) * T
dt = T / 400
tvd = lambda a, b: 0.5 * np.sum(np.abs(a - b))
tv_x, tv_p, margs = [], [], {}
xs, ps, tprev = Xf.ravel(), Pf.ravel(), 0.0
for t in ts:
    xs, ps = flow(xs, ps, t - tprev, dt)
    tprev = t
    cx = cic(xs, q, edges_x)
    cp = cic(ps, q, edges_p)
    qx, qp = marg_q(quantum_modes(t))
    tv_x.append(tvd(cx, qx)); tv_p.append(tvd(cp, qp))
    margs[t / T] = (cx, cp, qx, qp)
log("time series done")

snaps = {}
for s_ in SNAP:
    xb, pb = flow(Xw.ravel(), Pw.ravel(), -s_ * T, dt)
    wc = interp0(np.column_stack([xb, pb])).reshape(Xw.shape)
    wq = wigner_q(quantum_modes(s_ * T))
    snaps[s_] = (wc, wq)
    log(f"snapshot t = {s_} T done")

# ---------------------------------------------------------------- summary
tt = ts / T
lines = [f"Ground state of V = x^4/4 scaled by s = {S:g} in x and p (hbar = M = 1)",
         f"Quantum basis: {K} eigenstates below E = {E_CUT:g}; projected trace = {trace:.6f}",
         f"rho_0 is not positive: negative eigenvalues total {-neg_weight:.2e} "
         f"(most negative {neg_min:.2e})",
         f"<E> = {Emean:.1f},  classical period at <E>: T = {T:.4f}",
         f"check: max |w_0 - W_g(x/s,p/s)/s^2| / max w_0 = {w0_check:.2e}",
         "",
         "Instantaneous size of the dropped term at t = 0:",
         f"  ||(1/4) x w_ppp|| / ||p w_x||               = {ratio_px:.2e}",
         f"  ||(1/4) x w_ppp|| / ||w_t (classical)||     = {ratio_wt:.2e}",
         "  (unscaled ground state: 0.57 and 1, i.e. the whole w_t)",
         "",
         "Classical vs quantum, TV distance of the marginals (0 = identical):",
         "      t/T     position    momentum"]
for k in range(len(ts)):
    if k == 0 or k % 5 == 4 or k == len(ts) - 1:
        lines.append(f"  {tt[k]:8.2f}   {tv_x[k]:9.4f}   {tv_p[k]:9.4f}")
lines += ["",
          "Noise floor: the classical marginals are sampled from a finite lattice of",
          "trajectories.  Two classical runs (720^2 vs 1080^2 lattices) differ by TV",
          "0.004 / 0.012 (position / momentum) at t = 1 T and 0.027 / 0.025 at 10 T,",
          "so after a few periods the classical-vs-quantum numbers above are within",
          "the sampling noise: an upper bound, not a measured difference."]
summary = "\n".join(lines)
print(summary)
with open(os.path.join(OUT, "summary.txt"), "w") as fh:
    fh.write(summary + "\n")

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(2, len(SNAP), figsize=(3.3 * len(SNAP), 6.6), constrained_layout=True)
vmax = np.abs(w0).max()
for j, s_ in enumerate(SNAP):
    for i, (lab, w) in enumerate(zip(["classical", "quantum"], snaps[s_])):
        ax = axes[i, j]
        ax.imshow(w.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  extent=[xw[0], xw[-1], pw[0], pw[-1]], aspect="auto")
        ax.set_title(f"{lab}, t = {s_:g} T", fontsize=9)
        ax.set_xlabel("x"); ax.set_ylabel("p")
fig.suptitle(f"Ground state of x$^4$/4 scaled x{S:g}: classical Liouville vs quantum (T = {T:.3f})",
             fontsize=10)
fig.savefig(os.path.join(OUT, "snapshots.png"), dpi=130)

fig, axes = plt.subplots(2, len(SNAP), figsize=(3.3 * len(SNAP), 5.6), constrained_layout=True)
xc = 0.5 * (edges_x[1:] + edges_x[:-1]); pc = 0.5 * (edges_p[1:] + edges_p[:-1])
for j, s_ in enumerate(SNAP):
    key = min(margs, key=lambda k: abs(k - s_))
    cx, cp, qx, qp = margs[key]
    axes[0, j].plot(xc, qx / np.diff(edges_x), "k", lw=1.6, label="quantum")
    axes[0, j].plot(xc, cx / np.diff(edges_x), "C1--", lw=1.2, label="classical")
    axes[1, j].plot(pc, qp / np.diff(edges_p), "k", lw=1.6, label="quantum")
    axes[1, j].plot(pc, cp / np.diff(edges_p), "C1--", lw=1.2, label="classical")
    axes[0, j].set_title(f"t = {key:.3g} T", fontsize=9)
    axes[0, j].set_xlabel("x"); axes[1, j].set_xlabel("p")
axes[0, 0].set_ylabel("position density"); axes[1, 0].set_ylabel("momentum density")
axes[0, 0].legend(fontsize=8)
fig.savefig(os.path.join(OUT, "marginals.png"), dpi=130)

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax.loglog(tt[1:], tv_x[1:], "o-", ms=3, label="position marginal")
ax.loglog(tt[1:], tv_p[1:], "s-", ms=3, label="momentum marginal")
ax.set_xlabel("t / T (classical period at <E>)")
ax.set_ylabel("TV distance, classical vs quantum")
ax.set_title(f"Scaled ground state (x{S:g}) in x$^4$/4")
ax.legend()
fig.savefig(os.path.join(OUT, "drift_vs_t.png"), dpi=130)
