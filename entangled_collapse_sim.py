#!/usr/bin/env python3
"""
Collapse of a separated entangled pair as Liouville flow + conditioning.

Two free particles A and B (V constant) share an entangled state

    psi_AB = alpha |cat+>_A |L>_B + beta |cat->_A |R>_B ,

where cat+- = phi(x+a) +- phi(x-a) and L, R = phi(x+d), phi(x-d).
A "large" apparatus of N pointer modes couples ONLY to B by the impulsive
von Neumann interaction H = g(t) x_B sum_k P_k (integrated strength g = 1).

Reduction to one collective pointer (exact canonical change of variables):
  Xbar = mean_k X_k and P_tot = sum_k P_k are conjugate, H couples only to
  P_tot, and the N-1 relative modes decouple.  For N independent pointers of
  width sigma, the collective pointer is a minimum-uncertainty Gaussian of
  width sigma/sqrt(N).  "Large" means the collective readout becomes sharp.

Every Hamiltonian here is at most quadratic, so the Moyal correction
vanishes and classical Liouville flow of the signed Wigner function is
exact.  The classical side of this script never uses a wavefunction after
building w_0: it pushes signed phase-space weights through Hamilton's flow

    Xbar -> Xbar + x_B ,    p_B -> p_B - P_tot ,

reads the pointer (r = Xbar), and conditions on r > 0 or r < 0.

The quantum side is independent: it evolves psi(x_A, x_B, X) on a grid with
U = exp(-i x_B P_X), traces/projects, and computes Wigner functions.

Outputs (in collapse_sim_out/):
  conditional_wigner.png  - A's unconditional and conditional w, classical vs quantum
  vs_N.png                - outcome probability, AB coherence, A negativity vs N
  joint_momentum.png      - P(p_A, p_B) before/after the apparatus
  summary.txt             - numerical comparison table

Units: hbar = m = 1.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collapse_sim_out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- parameters
s = 0.5            # packet position width (|phi|^2 std)
a = 2.0            # A's cat half-separation
d = 2.0            # B's L/R half-separation
alpha2 = 0.7       # Born weight of the (cat+, L) branch
sigma1 = 6.0       # single-pointer width: one pointer cannot resolve 2d = 4
NS = [1, 4, 16, 64, 256]
N_SHOW = 64        # apparatus size used for the Wigner panels
SAMPLES = 24       # apparatus Monte Carlo samples per phase-space point
rng = np.random.default_rng(12345)

alpha, beta = np.sqrt(alpha2), np.sqrt(1 - alpha2)


def phi(x, c):
    return (2 * np.pi * s**2) ** -0.25 * np.exp(-(x - c) ** 2 / (4 * s**2))


def cat(x, sign):
    f = phi(x, -a) + sign * phi(x, a)
    return f / np.sqrt(2 * (1 + sign * np.exp(-a**2 / (2 * s**2))))


A_states = [lambda x: cat(x, +1), lambda x: cat(x, -1)]
B_states = [lambda x: phi(x, -d), lambda x: phi(x, +d)]
coef = np.array([alpha, beta])

# ------------------------------------------------- classical phase-space grid
xA = np.arange(-4.4, 4.4 + 1e-9, 0.2)
pA = np.arange(-5.0, 5.0 + 1e-9, 0.2)
xB = np.arange(-4.0, 4.0 + 1e-9, 0.2)
pB = np.arange(-5.0, 5.0 + 1e-9, 0.2)
dxA, dpA, dxB, dpB = 0.2, 0.2, 0.2, 0.2


def cross_wigner(f, g, x, p):
    """W[f,g](x,p) = (1/2pi) int dy e^{ipy} f*(x+y/2) g(x-y/2)  (paper's convention)."""
    y = np.linspace(-16, 16, 3201)
    dy = y[1] - y[0]
    F = np.conj(f(x[:, None] + y / 2)) * g(x[:, None] - y / 2)      # (nx, ny)
    return (F @ np.exp(1j * np.outer(y, p))) * dy / (2 * np.pi)       # (nx, np)


# w_AB = sum_ij c_i c_j W[A_i,A_j](xA,pA) W[B_i,B_j](xB,pB)  (real coefficients)
wAB = np.zeros((len(xA), len(pA), len(xB), len(pB)))
for i in range(2):
    for j in range(2):
        WA = cross_wigner(A_states[i], A_states[j], xA, pA)
        WB = cross_wigner(B_states[i], B_states[j], xB, pB)
        wAB += (coef[i] * coef[j] * np.einsum("ab,cd->abcd", WA, WB)).real
wdV = wAB * dxA * dpA * dxB * dpB
print(f"classical w_AB: sum = {wdV.sum():.6f}, min = {wAB.min():.4f}, "
      f"negative mass = {wdV[wdV < 0].sum():.4f}")

XB4 = np.broadcast_to(xB[None, None, :, None], wAB.shape).ravel()
PB4 = np.broadcast_to(pB[None, None, None, :], wAB.shape).ravel()
PA4 = np.broadcast_to(pA[None, :, None, None], wAB.shape).ravel()
W4 = wdV.ravel()

pB_edges = np.arange(-7.1, 7.1 + 1e-9, 0.2)
pB_hist = 0.5 * (pB_edges[1:] + pB_edges[:-1])
pA_edges = np.concatenate([pA - 0.1, [pA[-1] + 0.1]])


def classical_run(N):
    """Signed-weight Liouville flow through the apparatus, then conditioning."""
    sig = sigma1 / np.sqrt(N)
    cond_pos = np.zeros(len(xA) * len(pA))
    cond_neg = np.zeros_like(cond_pos)
    Pjoint = np.zeros((len(pA), len(pB_hist)))
    for _ in range(SAMPLES):
        X0 = rng.normal(0.0, sig, W4.size)              # collective pointer position
        Ptot = rng.normal(0.0, 1 / (2 * sig), W4.size)  # collective pointer momentum
        r = X0 + XB4                                    # Hamilton flow: Xbar += x_B
        pB_after = PB4 - Ptot                           #                p_B  -= P_tot
        pos = r > 0
        wpos = np.where(pos, W4, 0.0).reshape(len(xA) * len(pA), -1).sum(1)
        cond_pos += wpos
        cond_neg += W4.reshape(len(xA) * len(pA), -1).sum(1) - wpos
        H, _, _ = np.histogram2d(PA4, pB_after, bins=[pA_edges, pB_edges], weights=W4)
        Pjoint += H
    cond_pos /= SAMPLES
    cond_neg /= SAMPLES
    Pjoint /= SAMPLES * dpA * 0.2
    Ppos = cond_pos.sum()
    wA_pos = cond_pos.reshape(len(xA), len(pA)) / (Ppos * dxA * dpA)
    wA_neg = cond_neg.reshape(len(xA), len(pA)) / ((1 - Ppos) * dxA * dpA)
    return Ppos, wA_pos, wA_neg, Pjoint


wA_uncond_cl = wdV.sum(axis=(2, 3)) / (dxA * dpA)
P0_cl = wdV.sum(axis=(0, 2)) / (dpA * dpB)     # joint momentum before apparatus

# ----------------------------------------------------- independent quantum side
qxA = np.arange(-6.0, 6.0 + 1e-9, 0.1)
qxB = np.arange(-5.0, 5.0 + 1e-9, 0.1)
qdx = 0.1
psiAB = sum(coef[i] * np.outer(A_states[i](qxA), B_states[i](qxB)) for i in range(2))
psiAB /= np.sqrt((np.abs(psiAB) ** 2).sum() * qdx * qdx)


def wigner_from_rho(rho, x, p, dx):
    """W(x,p) = (1/2pi) int dy e^{ipy} rho(x - y/2, x + y/2), rho(x,x') = <x|rho|x'>."""
    n = len(x)
    W = np.zeros((n, len(p)))
    for i in range(n):
        kmax = min(i, n - 1 - i)
        k = np.arange(-kmax, kmax + 1)
        vals = rho[i - k, i + k]                                   # y = 2 k dx
        W[i] = (np.exp(1j * np.outer(p, 2 * k * dx)) @ vals).real * (2 * dx) / (2 * np.pi)
    return W


def dft_matrix(p, x, dx):
    return np.exp(-1j * np.outer(p, x)) * dx / np.sqrt(2 * np.pi)


MA = dft_matrix(pA, qxA, qdx)
MB = dft_matrix(pB_hist, qxB, qdx)


def quantum_run(N):
    sig = sigma1 / np.sqrt(N)
    L = 8 * sig + 12
    nX = 1024
    X = np.linspace(-L / 2, L / 2, nX, endpoint=False)
    dX = X[1] - X[0]
    chi = (2 * np.pi * sig**2) ** -0.25 * np.exp(-X**2 / (4 * sig**2))
    k = 2 * np.pi * np.fft.fftfreq(nX, dX)
    chik = np.fft.fft(chi)
    # U = exp(-i x_B P_X): pointer wavefunction shifted by x_B, for each x_B
    chi_shift = np.fft.ifft(chik[None, :] * np.exp(-1j * np.outer(qxB, k)), axis=1)  # (nxB, nX)
    pos = X > 0
    # rho_A conditioned on readout sign: sum over x_B and X in the readout region
    out = {}
    for name, mask in (("pos", pos), ("neg", ~pos)):
        wB = (np.abs(chi_shift[:, mask]) ** 2).sum(1) * dX          # P(readout in region | x_B)
        rho = (psiAB * wB[None, :]) @ psiAB.conj().T * qdx           # rho_A(x, x')
        prob = np.trace(rho).real * qdx
        out[name] = (prob, wigner_from_rho(rho / prob, qxA, pA, qdx))
    # joint momentum distribution, pointer traced out
    Pj = np.zeros((len(pA), len(pB_hist)))
    for iX in range(0, nX):
        if np.abs(chi_shift[:, iX]).max() < 1e-8:
            continue
        psi_X = psiAB * chi_shift[None, :, iX]
        Pj += np.abs(MA @ psi_X @ MB.T) ** 2 * dX
    return out["pos"][0], out["pos"][1], out["neg"][1], Pj


rhoA0 = psiAB @ psiAB.conj().T * qdx
wA_uncond_q = wigner_from_rho(rhoA0, qxA, pA, qdx)
sel = np.array([np.argmin(np.abs(qxA - v)) for v in xA])        # quantum rows at classical x_A
P0_q = np.abs(MA @ psiAB @ MB.T) ** 2

def fringe_amplitude(P, p_grid):
    """AB coherence seen in the joint momentum distribution.

    The A-B interference term oscillates in p_B as exp(+-i 2d p_B); the branch
    mixture is a smooth envelope.  Fourier-transforming in p_B at the conjugate
    separation u = 2d isolates the interference, and summing |.| over p_A avoids
    cancellation (the term is odd in p_A).  A kick distribution multiplies this
    amplitude by its characteristic function, exp(-var_kick (2d)^2 / 2).
    """
    u = 2 * d
    return float(np.abs(P @ np.exp(1j * u * p_grid)).sum())


HIST_SINC = np.sin(0.1 * 2 * d) / (0.1 * 2 * d)   # 0.2-wide histogram bins smooth fringes


# ---------------------------------------------------------------------- runs
rows = []
store = {}
for N in NS:
    cl = classical_run(N)
    qu = quantum_run(N)
    store[N] = (cl, qu)
    ref = fringe_amplitude(P0_q, pB_hist)
    coh_cl = fringe_amplitude(cl[3], pB_hist) / (ref * HIST_SINC)
    coh_q = fringe_amplitude(qu[3], pB_hist) / ref
    coh_th = np.exp(-N * (2 * d) ** 2 / (8 * sigma1**2))
    err_pos = np.abs(cl[1] - qu[1][sel]).max()
    err_neg = np.abs(cl[2] - qu[2][sel]).max()
    rows.append((N, sigma1 / np.sqrt(N), cl[0], qu[0], coh_cl, coh_q, coh_th,
                 cl[1].min(), qu[1].min(), err_pos, err_neg))
    print(f"N={N:4d} done")

hdr = ("   N  ptr_width  P(r>0)_cl  P(r>0)_qm  coh_cl  coh_qm  coh_theory"
       "  min wA|r>0 cl   qm   max|dw| r>0  r<0")
lines = [hdr]
for r in rows:
    lines.append(f"{r[0]:4d}  {r[1]:8.3f}  {r[2]:9.4f}  {r[3]:9.4f}  {r[4]:6.3f}  {r[5]:6.3f}"
                 f"  {r[6]:9.3f}   {r[7]:8.4f} {r[8]:8.4f}   {r[9]:9.4f} {r[10]:7.4f}")
nosig = np.abs(wA_uncond_cl - wA_uncond_q[sel]).max()
lines.append(f"\nA unconditional: max |w_cl - w_qm| = {nosig:.4f}; "
             f"min w_A uncond (qm) = {wA_uncond_q.min():.4f}")
(clN, quN) = store[N_SHOW]
mix = clN[0] * clN[1] + (1 - clN[0]) * clN[2]
lines.append(f"No signaling (N={N_SHOW}): max |P+ w_A|+ + P- w_A|- - w_A uncond| = "
             f"{np.abs(mix - wA_uncond_cl).max():.2e}")
summary = "\n".join(lines)
print(summary)
with open(os.path.join(OUT, "summary.txt"), "w") as fh:
    fh.write(summary + "\n")

# -------------------------------------------------------------------- figures
ext = [pA[0], pA[-1], xA[0], xA[-1]]
panels = [("unconditional", wA_uncond_cl, wA_uncond_q[sel]),
          ("conditioned on r > 0", clN[1], quN[1][sel]),
          ("conditioned on r < 0", clN[2], quN[2][sel])]
vmax = max(np.abs(p[2]).max() for p in panels)
fig, ax = plt.subplots(3, 3, figsize=(13, 12), layout="constrained")
for j, (title, wc, wq) in enumerate(panels):
    for i, (lab, w) in enumerate((("classical Liouville + conditioning", wc), ("quantum", wq))):
        im = ax[i, j].imshow(w, origin="lower", extent=ext, aspect="auto",
                             cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax[i, j].set_title(f"$w_A$ {title}\n{lab}", fontsize=10)
        ax[i, j].set_xlabel("$p_A$")
        ax[i, j].set_ylabel("$x_A$")
    i0 = np.argmin(np.abs(xA))
    ax[2, j].plot(pA, wq[i0], "k-", lw=2, label="quantum")
    ax[2, j].plot(pA, wc[i0], "o", ms=4, color="C1", label="classical")
    ax[2, j].axhline(0, color="gray", lw=0.5)
    ax[2, j].set_title(f"cut at $x_A = 0$ ({title})", fontsize=10)
    ax[2, j].set_xlabel("$p_A$")
    ax[2, j].legend(fontsize=8)
fig.colorbar(im, ax=ax[:2, :], shrink=0.8)
fig.suptitle(f"A's phase-space distribution; apparatus of N = {N_SHOW} pointers coupled only to B",
             fontsize=12)
fig.savefig(os.path.join(OUT, "conditional_wigner.png"), dpi=130)
plt.close(fig)

R = np.array(rows)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].semilogx(R[:, 0], R[:, 3], "k-", lw=2, label="quantum")
ax[0].semilogx(R[:, 0], R[:, 2], "o", color="C1", label="classical")
ax[0].axhline(1 - alpha2, ls="--", color="gray", label=r"Born $|\beta|^2$ (B at $+d$)")
ax[0].set_title("P(readout r > 0)")
ax[1].semilogx(R[:, 0], R[:, 5], "k-", lw=2, label="quantum")
ax[1].semilogx(R[:, 0], R[:, 4], "o", color="C1", label="classical")
ax[1].semilogx(R[:, 0], R[:, 6], "--", color="gray", label=r"$e^{-N(2d)^2/8\sigma^2}$")
ax[1].set_title("A-B coherence (momentum fringes)")
ax[2].semilogx(R[:, 0], R[:, 8], "k-", lw=2, label="quantum")
ax[2].semilogx(R[:, 0], R[:, 7], "o", color="C1", label="classical")
ax[2].axhline(wA_uncond_q.min(), ls="--", color="gray", label="unconditional")
ax[2].set_title(r"min $w_A$ given r > 0 (negativity)")
for x_ in ax:
    x_.set_xlabel("apparatus size N")
    x_.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "vs_N.png"), dpi=130)
plt.close(fig)

fig, ax = plt.subplots(2, 2, figsize=(10, 9))
extp = [pB_hist[0], pB_hist[-1], pA[0], pA[-1]]
Nmax = NS[-1]
cols = (("before apparatus", P0_cl, [pB[0], pB[-1], pA[0], pA[-1]], P0_q),
        (f"after N = {Nmax}", store[Nmax][0][3], extp, store[Nmax][1][3]))
for j, (lab, Pc, extc, Pq) in enumerate(cols):
    ax[0, j].imshow(Pc, origin="lower", extent=extc, aspect="auto", cmap="viridis")
    ax[0, j].set_title(f"classical $P(p_A,p_B)$ {lab}", fontsize=10)
    ax[1, j].imshow(Pq, origin="lower", extent=extp, aspect="auto", cmap="viridis")
    ax[1, j].set_title(f"quantum $P(p_A,p_B)$ {lab}", fontsize=10)
    for i in range(2):
        ax[i, j].set_xlabel("$p_B$")
        ax[i, j].set_ylabel("$p_A$")
        ax[i, j].set_xlim(-6, 6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "joint_momentum.png"), dpi=130)
plt.close(fig)
print(f"figures written to {OUT}")
