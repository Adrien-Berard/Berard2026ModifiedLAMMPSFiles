"""
Publication-quality dynamical systems analysis of chromatin–condensate coupling.
Five-panel figure (a–e): hysteresis, bifurcation, phase plane, stability map,
basin of attraction.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.optimize import fsolve, brentq
from scipy.integrate import solve_ivp
from scipy.linalg import eigvals as sp_eigvals

# ── rcParams ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif'],
    'mathtext.fontset':  'dejavuserif',
    'axes.linewidth':    0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size':  3.2,
    'ytick.major.size':  3.2,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'lines.linewidth':   1.5,
    'font.size':         8,
    'axes.labelsize':    9,
    'axes.titlesize':    9,
    'legend.fontsize':   7.5,
    'figure.dpi':        200,
})

# ── Model parameters (tuned for clear bistability) ──────────────────────────
k1    = 1
# k2    = 2
Gamma = 0.4
gamma = 25.0
rho   = 60.0
N     = 80.0
beta  = 4.0
C_N   = 5.0
C_s   = 3.0

def V(s):
    return C_N * N - C_s * s

def dm_dt(m, s, k2):
    return k2**2 * s * m*(1-m) - k1*k2 * m*(1-m)**2 + Gamma*(1-2*m)

def ds_dt(m, s):
    Vs = V(s)
    if Vs <= 0.01:
        return 0.0
    phi = m * N / Vs
    term1 = rho / Vs**(2/3) * phi * (1 - phi - s)
    term2 = s * np.exp(-beta * (phi + s))
    return gamma * (term1 - term2)

def jacobian(m, s, k2, eps=1e-6):
    F = lambda x: dm_dt(x[0], x[1], k2)
    G = lambda x: ds_dt(x[0], x[1])
    x = np.array([m, s], dtype=float)
    J = np.zeros((2, 2))
    for i in range(2):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps; xm[i] -= eps
        J[0, i] = (F(xp) - F(xm)) / (2*eps)
        J[1, i] = (G(xp) - G(xm)) / (2*eps)
    return J

def classify(J):
    det = np.linalg.det(J)
    tr  = np.trace(J)
    if det < 0:    return 'saddle'
    elif tr < 0:   return 'stable'
    else:          return 'unstable'

def find_fps(k2, n_m=50, n_s=25, s_max=1.2):
    """Return list of (m, s) fixed points at given k2."""
    def res(x):
        m, s = x
        if m <= 0 or m >= 1 or s < 0 or V(s) <= 0.01:
            return [1e6, 1e6]
        return [dm_dt(m, s, k2), ds_dt(m, s)]
    fps = []
    for m0 in np.linspace(0.03, 0.97, n_m):
        for s0 in np.linspace(0.005, s_max, n_s):
            try:
                sol = fsolve(res, [m0, s0], full_output=True)
                x, _, ier, _ = sol
                if ier != 1:
                    continue
                m_fp, s_fp = x
                if not (0.008 < m_fp < 0.992 and 0.0005 < s_fp and V(s_fp) > 0.05):
                    continue
                if np.max(np.abs(res(x))) > 1e-6:
                    continue
                dup = any(abs(fp[0]-m_fp) < 0.02 and abs(fp[1]-s_fp) < 0.02
                          for fp in fps)
                if not dup:
                    fps.append((float(m_fp), float(s_fp)))
            except Exception:
                pass
    return fps

# ── Colour palette ───────────────────────────────────────────────────────────
C_STABLE   = '#1a6f9e'   # steel blue
C_UNSTABLE = '#c94040'   # crimson
C_SADDLE   = '#d4860a'   # amber
C_NC_M     = '#2a9d5a'   # green  (m-nullcline)
C_NC_S     = '#7b4ba0'   # violet (s-nullcline)
C_ATT0     = '#1a6f9e'   # basin 0
C_ATT1     = '#c07020'   # basin 1

K2_REF = 2.0
# K2_VALS = np.linspace(1/4, 4, 140)
K2_VALS = np.logspace(np.log10(0.25), np.log10(4), 140)

# ════════════════════════════════════════════════════════════════════════════
print("1/5  Computing bifurcation diagram …")
bif_stable   = {'k2': [], 'm': []}
bif_saddle   = {'k2': [], 'm': []}
bif_unstable = {'k2': [], 'm': []}

for k2 in K2_VALS:
    for m_fp, s_fp in find_fps(k2):
        J  = jacobian(m_fp, s_fp, k2)
        st = classify(J)
        if st == 'stable':
            bif_stable['k2'].append(k2);   bif_stable['m'].append(m_fp)
        elif st == 'saddle':
            bif_saddle['k2'].append(k2);   bif_saddle['m'].append(m_fp)
        else:
            bif_unstable['k2'].append(k2); bif_unstable['m'].append(m_fp)

print("2/5  Computing hysteresis sweep …")
m_fwd, m_bwd = [], []
m_cur = 0.05
for k2 in K2_VALS:
    fps = [(m, s) for m, s in find_fps(k2)
           if classify(jacobian(m, s, k2)) == 'stable']
    if not fps:
        m_fwd.append(np.nan); continue
    best = min(fps, key=lambda x: abs(x[0] - m_cur))
    m_cur = best[0]; m_fwd.append(m_cur)

m_cur = 0.95
for k2 in reversed(K2_VALS):
    fps = [(m, s) for m, s in find_fps(k2)
           if classify(jacobian(m, s, k2)) == 'stable']
    if not fps:
        m_bwd.append(np.nan); continue
    best = min(fps, key=lambda x: abs(x[0] - m_cur))
    m_cur = best[0]; m_bwd.append(m_cur)
m_bwd = list(reversed(m_bwd))
m_fwd = np.array(m_fwd); m_bwd = np.array(m_bwd)

print("3/5  Computing phase-plane structure …")
fps_ref = find_fps(K2_REF)
m_arr   = np.linspace(0.02, 0.98, 400)

# m-nullcline: dm/dt = 0  => s = [k1*k2*m*(1-m)^2 - Gamma*(1-2m)] / [k2^2*m*(1-m)]
def m_nc_s(m, k2):
    denom = k2**2 * m * (1 - m)
    if abs(denom) < 1e-12:
        return np.nan
    return (k1*k2 * m*(1-m)**2 - Gamma*(1-2*m)) / denom

s_mnc = np.array([m_nc_s(m, K2_REF) for m in m_arr])
valid_mnc = np.isfinite(s_mnc) & (s_mnc > 0) & (V(s_mnc) > 0.05)

# s-nullcline: ds/dt = 0 at each m, solve for s numerically
s_snc = np.full(len(m_arr), np.nan)
for idx, m in enumerate(m_arr):
    if m < 0.03 or m > 0.97:
        continue
    try:
        f = lambda s: ds_dt(m, s) if V(s) > 0.01 else 1.0
        # check sign change
        sv = np.linspace(0.005, 1.0, 300)
        fv = [f(s) for s in sv]
        crosses = [(sv[i], sv[i+1]) for i in range(len(fv)-1)
                   if fv[i]*fv[i+1] < 0]
        if crosses:
            s_sol = brentq(f, crosses[0][0], crosses[0][1], maxiter=100)
            s_snc[idx] = s_sol
    except Exception:
        pass
valid_snc = np.isfinite(s_snc)

print("4/5  Computing stability map …")
m_grid = np.linspace(0.03, 0.97, 200)
s_grid = np.linspace(0.005, 1.2, 160)
MG, SG = np.meshgrid(m_grid, s_grid)
SMAP   = np.full(MG.shape, np.nan)
TRMAP  = np.full(MG.shape, np.nan)
DETMAP = np.full(MG.shape, np.nan)

for i in range(MG.shape[0]):
    for j in range(MG.shape[1]):
        m0, s0 = MG[i, j], SG[i, j]
        if V(s0) <= 0.05:
            continue
        J = jacobian(m0, s0, K2_REF)
        tr  = np.trace(J)
        det = np.linalg.det(J)
        TRMAP[i, j]  = tr
        DETMAP[i, j] = det
        if det < 0:
            SMAP[i, j] = 0     # saddle
        elif tr < 0:
            SMAP[i, j] = 1     # stable
        else:
            SMAP[i, j] = 2     # unstable

# print("5/5  Computing basin of attraction …")
# stable_fps = [(m, s) for m, s in fps_ref
#               if classify(jacobian(m, s, K2_REF)) == 'stable']

# m_ic = np.linspace(0.03, 0.97, 90)
# s_ic = np.linspace(0.005, 1.2, 72)
# MIC, SIC = np.meshgrid(m_ic, s_ic)
# BASIN = np.full(MIC.shape, np.nan)

# def ode_sys(t, y):
#     m, s = y
#     if m <= 0 or m >= 1 or s < 0 or V(s) <= 0.01:
#         return [0.0, 0.0]
#     return [dm_dt(m, s, K2_REF), ds_dt(m, s)]

# for i in range(MIC.shape[0]):
#     for j in range(MIC.shape[1]):
#         m0, s0 = MIC[i, j], SIC[i, j]
#         if V(s0) <= 0.05:
#             continue
#         try:
#             sol = solve_ivp(ode_sys, [0, 300], [m0, s0],
#                             method='RK45', max_step=0.5,
#                             rtol=1e-5, atol=1e-7, dense_output=False)
#             m_end, s_end = sol.y[0, -1], sol.y[1, -1]
#             if not stable_fps:
#                 BASIN[i, j] = 0
#                 continue
#             dists = [np.hypot(m_end - fp[0], s_end - fp[1])
#                      for fp in stable_fps]
#             BASIN[i, j] = float(np.argmin(dists))
#         except Exception:
#             pass

# ════════════════════════════════════════════════════════════════════════════
print("Rendering figure …")

fig = plt.figure(figsize=(15, 6.2))
gs  = fig.add_gridspec(2, 6,
                        left=0.06, right=0.97,
                        bottom=0.11, top=0.91,
                        hspace=0.60, wspace=0.70)

ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[0, 4:6])
ax_d = fig.add_subplot(gs[1, 0:3])
ax_e = fig.add_subplot(gs[1, 3:6])

for label, ax in zip('abcde', [ax_a, ax_b, ax_c, ax_d, ax_e]):
    ax.text(-0.14, 1.07, f'({label})',
            transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left')

# ── (a) Hysteresis ───────────────────────────────────────────────────────────
ax = ax_a
cmap_h = plt.get_cmap('RdBu_r')
norm_h = mcolors.Normalize(K2_VALS.min(), K2_VALS.max())

# for i in range(len(K2_VALS) - 1):
#     if not np.isnan(m_fwd[i]) and not np.isnan(m_fwd[i+1]):
#         ax.plot(K2_VALS[i:i+2], m_fwd[i:i+2],
#                 color=cmap_h(norm_h(K2_VALS[i])), lw=2.2, solid_capstyle='round')
#     if not np.isnan(m_bwd[i]) and not np.isnan(m_bwd[i+1]):
#         ax.plot(K2_VALS[i:i+2], m_bwd[i:i+2], lw=2.2,
#                 linestyle=(0, (5, 2.5)), solid_capstyle='round')

for i in range(len(K2_VALS) - 1):
    if not np.isnan(m_fwd[i]) and not np.isnan(m_fwd[i+1]):
        ax.plot(K2_VALS[i:i+2], m_fwd[i:i+2],
                color=cmap_h(norm_h(K2_VALS[i])),
                lw=2.2,
                solid_capstyle='round')

    if not np.isnan(m_bwd[i]) and not np.isnan(m_bwd[i+1]):
        ax.plot(K2_VALS[i:i+2], m_bwd[i:i+2],
                color='black',   # or '#555555'
                lw=2.0,
                linestyle='--',
                alpha=0.8,
                solid_capstyle='round')
        
sm = plt.cm.ScalarMappable(cmap=cmap_h, norm=norm_h)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, pad=0.04, fraction=0.045, shrink=0.9)
cb.set_label(r'$k_2$', fontsize=8)
cb.ax.tick_params(labelsize=7)
cb.set_ticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

# direction arrows
ax.annotate('', xy=(1.5, 0.08), xytext=(1.1, 0.06),
            arrowprops=dict(arrowstyle='->', color='#3366cc', lw=1.2))
ax.annotate('', xy=(2.8, 0.85), xytext=(3.2, 0.88),
            arrowprops=dict(arrowstyle='->', color='#cc3333', lw=1.2))
ax.text(1.05, 0.035, 'forward', fontsize=7, color='#3366cc')
ax.text(2.6,  0.91,  'backward', fontsize=7, color='#cc3333')

ax.set_xlabel(r'Control parameter $k_2$')
ax.set_ylabel(r'Steady-state $m^*$', labelpad=2)
ax.set_title('Hysteresis loop', pad=5)
ax.set_xlim(K2_VALS[0], K2_VALS[-1])
ax.set_ylim(-0.02, 1.05)
ax.set_xscale('log')

# ── (b) Bifurcation diagram ──────────────────────────────────────────────────
ax = ax_b

if bif_stable['k2']:
    ax.scatter(bif_stable['k2'],   bif_stable['m'],   s=2.5,
               color=C_STABLE,   linewidths=0, zorder=4, label='Stable')
if bif_saddle['k2']:
    ax.scatter(bif_saddle['k2'],   bif_saddle['m'],   s=2.5,
               color=C_SADDLE,   linewidths=0, zorder=4, label='Saddle')
if bif_unstable['k2']:
    ax.scatter(bif_unstable['k2'], bif_unstable['m'], s=2.5,
               color=C_UNSTABLE, linewidths=0, zorder=4, label='Unstable')

# shade bistable region
from collections import defaultdict
k2_stable_ms = defaultdict(list)
for k2, m in zip(bif_stable['k2'], bif_stable['m']):
    k2_stable_ms[round(k2, 4)].append(m)
bistable_k2 = [k2 for k2, ms in k2_stable_ms.items() if len(ms) >= 2]
if bistable_k2:
    k2_lo, k2_hi = min(bistable_k2), max(bistable_k2)
    ax.axvspan(k2_lo, k2_hi, alpha=0.10, color=C_STABLE, zorder=0,
               label='Bistable region')
    ax.axvline(k2_lo, color=C_STABLE, lw=0.8, ls=':', alpha=0.7)
    ax.axvline(k2_hi, color=C_STABLE, lw=0.8, ls=':', alpha=0.7)
    ax.text(k2_lo + 0.04, 0.96, r'$k_2^{(1)}$', fontsize=7.5,
            color=C_STABLE, va='top')
    ax.text(k2_hi + 0.04, 0.96, r'$k_2^{(2)}$', fontsize=7.5,
            color=C_STABLE, va='top')

ax.set_xlabel(r'Control parameter $k_2$')
ax.set_ylabel(r'Fixed point $m^*$', labelpad=2)
ax.set_title('Bifurcation diagram', pad=5)
ax.set_xlim(K2_VALS[0], K2_VALS[-1])
ax.set_ylim(-0.02, 1.05)
ax.legend(frameon=False, loc='upper left',
          markerscale=3, handlelength=1.0, borderpad=0.3)

# ── (c) Phase plane ──────────────────────────────────────────────────────────
ax = ax_c

# nullclines
ax.plot(m_arr[valid_mnc], s_mnc[valid_mnc],
        color=C_NC_M, lw=1.8, zorder=4, label=r'$\dot{m}=0$')
ax.plot(m_arr[valid_snc], s_snc[valid_snc],
        color=C_NC_S, lw=1.8, zorder=4, label=r'$\dot{s}=0$')

# vector field
mg, sg = np.meshgrid(np.linspace(0.06, 0.94, 20),
                      np.linspace(0.01, 1.15, 16))
U = np.zeros_like(mg); W = np.zeros_like(sg)
for i in range(mg.shape[0]):
    for j in range(mg.shape[1]):
        m0, s0 = mg[i,j], sg[i,j]
        if V(s0) > 0.05:
            U[i,j] = dm_dt(m0, s0, K2_REF)
            W[i,j] = ds_dt(m0, s0)
mag = np.sqrt(U**2 + W**2) + 1e-12
ax.quiver(mg, sg, U/mag, W/mag,
          alpha=0.35, scale=26, width=0.0030,
          headwidth=3.5, headlength=4.5, color='#555555', zorder=2)

# fixed points
for m_fp, s_fp in fps_ref:
    J  = jacobian(m_fp, s_fp, K2_REF)
    st = classify(J)
    if st == 'stable':
        ax.plot(m_fp, s_fp, 'o', ms=8, color=C_STABLE, zorder=6,
                mec='white', mew=1.0)
    elif st == 'saddle':
        ax.plot(m_fp, s_fp, 's', ms=7, color=C_SADDLE, zorder=6,
                mec='white', mew=1.0)
    else:
        ax.plot(m_fp, s_fp, 'o', ms=8, color=C_UNSTABLE, zorder=6,
                mfc='none', mew=1.5)

ax.set_xlabel(r'Chromatin occupancy $m$')
ax.set_ylabel(r'Condensate fraction $s$', labelpad=2)
ax.set_title(fr'Phase plane  ($k_2 = {K2_REF}$)', pad=5)
ax.set_xlim(0.02, 0.98); ax.set_ylim(0, 1.25)

handles_c = [
    Line2D([0],[0], color=C_NC_M, lw=1.8),
    Line2D([0],[0], color=C_NC_S, lw=1.8),
    Line2D([0],[0], marker='o', color=C_STABLE,   ls='none', ms=7,
           mec='white', mew=0.8),
    Line2D([0],[0], marker='s', color=C_SADDLE,   ls='none', ms=6,
           mec='white', mew=0.8),
]
ax.legend(handles_c,
          [r'$\dot{m}=0$', r'$\dot{s}=0$', 'Stable node', 'Saddle'],
          frameon=False, loc='upper right',
          handlelength=1.2, borderpad=0.3)

# ── (d) Stability map ────────────────────────────────────────────────────────
ax = ax_d

cmap_st  = mcolors.ListedColormap([C_SADDLE, C_STABLE, C_UNSTABLE])
bounds_st = [-0.5, 0.5, 1.5, 2.5]
norm_st  = mcolors.BoundaryNorm(bounds_st, cmap_st.N)

im = ax.pcolormesh(MG, SG, SMAP,
                   cmap=cmap_st, norm=norm_st,
                   alpha=0.50, shading='auto', rasterized=True)

# tr = 0 and det = 0 contours
cs_tr  = ax.contour(MG, SG, TRMAP,  levels=[0],
                    colors=['#111111'], linewidths=0.9,
                    linestyles='--', alpha=0.75)
cs_det = ax.contour(MG, SG, DETMAP, levels=[0],
                    colors=['#111111'], linewidths=0.9,
                    linestyles=':', alpha=0.75)
ax.clabel(cs_tr,  fmt=r'$\mathrm{tr}=0$',  fontsize=6.5, inline=True,
          manual=False)
ax.clabel(cs_det, fmt=r'$\det=0$',          fontsize=6.5, inline=True,
          manual=False)

# fixed points
for m_fp, s_fp in fps_ref:
    J  = jacobian(m_fp, s_fp, K2_REF)
    st = classify(J)
    col = {'stable': C_STABLE, 'saddle': C_SADDLE, 'unstable': C_UNSTABLE}[st]
    ax.plot(m_fp, s_fp, '*', ms=11, color=col, zorder=7,
            mec='white', mew=0.8)

cb2 = fig.colorbar(im, ax=ax, pad=0.03, fraction=0.025, shrink=0.85)
cb2.set_ticks([0, 1, 2])
cb2.set_ticklabels(['Saddle', 'Stable', 'Unstable'], fontsize=7)

ax.set_xlabel(r'Chromatin occupancy $m$')
ax.set_ylabel(r'Condensate fraction $s$', labelpad=2)
ax.set_title(fr'Stability classification  ($k_2 = {K2_REF}$)', pad=5)
ax.set_xlim(0.03, 0.97); ax.set_ylim(0.005, 1.2)

legend_d = [
    Line2D([0],[0], color='#111111', lw=0.9, ls='--',
           label=r'$\mathrm{tr}(J)=0$'),
    Line2D([0],[0], color='#111111', lw=0.9, ls=':',
           label=r'$\det(J)=0$'),
    Line2D([0],[0], marker='*', color=C_STABLE,  ls='none', ms=9,
           mec='white', mew=0.6, label='Stable node'),
    Line2D([0],[0], marker='*', color=C_SADDLE,  ls='none', ms=9,
           mec='white', mew=0.6, label='Saddle'),
]
ax.legend(handles=legend_d, frameon=False, loc='upper right',
          handlelength=1.5, borderpad=0.3, ncol=2)

# ── (e) Basin of attraction ──────────────────────────────────────────────────
# ax = ax_e

# n_att = len(stable_fps)
# if n_att >= 2:
#     basin_cmap = mcolors.ListedColormap([C_ATT0, C_ATT1])
#     basin_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], 2)
# elif n_att == 1:
#     basin_cmap = mcolors.ListedColormap([C_ATT0])
#     basin_norm = mcolors.BoundaryNorm([-0.5, 0.5], 1)
# else:
#     basin_cmap = 'viridis'; basin_norm = None

# ax.pcolormesh(MIC, SIC, BASIN,
#               cmap=basin_cmap, norm=basin_norm,
#               alpha=0.50, shading='auto', rasterized=True)

# # overlay nullclines as separatrix guide
# ax.plot(m_arr[valid_mnc], s_mnc[valid_mnc],
#         color=C_NC_M, lw=1.1, alpha=0.65, ls='--')
# ax.plot(m_arr[valid_snc], s_snc[valid_snc],
#         color=C_NC_S, lw=1.1, alpha=0.65, ls='--')

# # attractors
# att_colors = [C_ATT0, C_ATT1]
# for k, (m_fp, s_fp) in enumerate(stable_fps):
#     col = att_colors[k % 2]
#     ax.plot(m_fp, s_fp, '*', ms=12, color=col, zorder=6,
#             mec='white', mew=0.8)
# # saddle points
# for m_fp, s_fp in fps_ref:
#     if classify(jacobian(m_fp, s_fp, K2_REF)) == 'saddle':
#         ax.plot(m_fp, s_fp, 's', ms=7, color=C_SADDLE, zorder=6,
#                 mec='white', mew=0.8)

# ax.set_xlabel(r'Chromatin occupancy $m$')
# ax.set_ylabel(r'Condensate fraction $s$', labelpad=2)
# ax.set_title(fr'Basin of attraction  ($k_2 = {K2_REF}$)', pad=5)
# ax.set_xlim(0.03, 0.97); ax.set_ylim(0.005, 1.2)

# legend_e = []
# for k, (m_fp, s_fp) in enumerate(stable_fps):
#     legend_e.append(
#         Line2D([0],[0], marker='*', color=att_colors[k%2], ls='none', ms=9,
#                mec='white', mew=0.6,
#                label=fr'Attractor {k+1}  ($m^*={m_fp:.2f}$)'))
# legend_e.append(
#     Line2D([0],[0], marker='s', color=C_SADDLE, ls='none', ms=6,
#            mec='white', mew=0.6, label='Saddle (separatrix)'))
# legend_e += [
#     Line2D([0],[0], color=C_NC_M, lw=1.0, ls='--', alpha=0.7,
#            label=r'$\dot{m}=0$'),
#     Line2D([0],[0], color=C_NC_S, lw=1.0, ls='--', alpha=0.7,
#            label=r'$\dot{s}=0$'),
# ]
# ax.legend(handles=legend_e, frameon=False, loc='upper right',
#           handlelength=1.4, borderpad=0.3)

# # ── Suptitle ─────────────────────────────────────────────────────────────────
# fig.suptitle(
#     r'Chromatin–Condensate Dynamical System: Steady-State Analysis',
#     fontsize=10.5, fontweight='bold', y=0.97)

# ── Save ──────────────────────────────────────────────────────────────────────
out_pdf = 'outputs/chromatin_condensate_analysis.pdf'
out_png = 'outputs/chromatin_condensate_analysis.png'
fig.savefig(out_pdf, dpi=250, bbox_inches='tight', facecolor='white')
fig.savefig(out_png, dpi=250, bbox_inches='tight', facecolor='white')
print(f"Saved:\n  {out_pdf}\n  {out_png}")
plt.close(fig)
