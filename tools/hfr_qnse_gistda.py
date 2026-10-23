#!/usr/bin/env python3
"""hfr_qnse_gistda.py — QNSE comparison for the GISTDA HF radar totals (upper Gulf of Thailand,
13.1°N): normalisation check, parameter-free f-branch overlay, C_L/C_T/R diagnostics with block
bootstrap, configuration-separated zonal spectra, time-filter sensitivity, Bühler sensitivity,
large-ensemble validation and a bottom-friction estimate.

Spectral convention (all spectra here):
  * one-sided wavenumber spectrum, k in cycles/km (also given in rad/m), E in (m/s)² per cycle/km
    (rad/m: m³/s², E_rad = E_cpkm · 1000/2π, k_rad = 2π k_cpkm/1000)
  * per segment: de-meaned (no detrend), Tukey(0.5) taper, |FFT|² · Δx / (L · ⟨w²⟩), doubled except
    DC and Nyquist ⇒ Σ E Δk = ⟨(d·w)²⟩/⟨w²⟩ ≈ ⟨d²⟩ (taper loss corrected) — verified per group
  * zonal transects only (like-with-like between radar configurations), 26-node (52 km) segments,
    interior gaps ≤ 2 nodes filled, ≤ 15 % filled per segment

QNSE (Galperin & Sukoriansky 2020, Phys. Rev. Fluids 5, 063803; IOVWST 2022 slides):
  E_L = (18/55) C_K Π^(2/3) k^-5/3 + 0.0926 f² k^-3,  E_T = (24/55) C_K Π^(2/3) k^-5/3 + 0.24 f² k^-3
  (k in rad/m). The f-branch is parameter-free; where observations lie below it, no Π ≥ 0 can fit.

Filters compared: (a) UTide detiding (auto constituents, Rayleigh = 1, per node) + clockwise inertial
notch by complex demodulation, no low-pass; (b) Lanczos 72 h low-pass.

Example:
  python tools/hfr_qnse_gistda.py --realisations 50
"""
import argparse
import json
import math
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.signal.windows import tukey

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hfr_spectra as S  # noqa: E402
import hfr_validate as V  # noqa: E402

OMEGA = 7.2921e-5
LAT = 13.1
F = 2 * OMEGA * math.sin(math.radians(LAT))
QL, QT = 0.0926, 0.24
C_K = 1.6
SEG, MAXFILL, FLOOR_KM = 26, 2, 20.0
BLOCK_H = 168                       # bootstrap block: 7 days (subtidal fields stay correlated for days)
CD, H_DEPTHS = 2.5e-3, (10.0, 15.0, 25.0)
T_INERTIAL_H = 2 * math.pi / F / 3600

# ───────────────────────── time filters ─────────────────────────

def utide_detide(u, v, t_epoch, min_frac=0.6):
    """Per-node UTide (OLS, automatic constituent selection, Rayleigh 1). Returns residuals and the
    reconstructed tidal currents."""
    import utide
    nt, ny, nx = u.shape
    t = (t_epoch * 1e9).astype('datetime64[ns]')
    ru, rv = np.full_like(u, np.nan), np.full_like(v, np.nan)
    tu, tv = np.full_like(u, np.nan), np.full_like(v, np.nan)
    for j in range(ny):
        for i in range(nx):
            ok = np.isfinite(u[:, j, i]) & np.isfinite(v[:, j, i])
            if ok.mean() < min_frac:
                continue
            c = utide.solve(t[ok], u[ok, j, i], v[ok, j, i], lat=LAT, method='ols', conf_int='none',
                            trend=False, verbose=False)
            r = utide.reconstruct(t[ok], c, verbose=False)
            tide_u, tide_v = r.u - c.umean, r.v - c.vmean          # tidal part only (keep the mean flow)
            tu[ok, j, i], tv[ok, j, i] = tide_u, tide_v
            ru[ok, j, i], rv[ok, j, i] = u[ok, j, i] - tide_u, v[ok, j, i] - tide_v
    return ru, rv, tu, tv


def inertial_notch(u, v, cutoff_h=120.0, half_h=120.0):
    """Remove the clockwise (NH) near-inertial band: demodulate w = u + iv at −f, low-pass the
    envelope (Lanczos, cutoff_h), remodulate and subtract. NaN-aware; ends with too little data → NaN."""
    nt = u.shape[0]
    t = np.arange(nt) * 3600.0
    ph = np.exp(1j * F * t)[:, None, None]                    # w·e^{+ift} shifts clockwise −f to 0
    w = (u + 1j * v) * ph
    lw = S.lanczos_weights(cutoff_h, half_h, 1.0)
    env = S.lowpass(w.real, lw) + 1j * S.lowpass(w.imag, lw)
    inert = env / ph
    return u - inert.real, v - inert.imag


def filtered(path, kind):
    d = np.load(path)
    u, v = d['u'].astype(float), d['v'].astype(float)
    out = {}
    if kind == 'lanczos':
        w = S.lanczos_weights(72.0, 120.0, 1.0)
        u, v = S.lowpass(u, w), S.lowpass(v, w)
    elif kind == 'utide_notch':
        ru, rv, tu, tv = utide_detide(u, v, d['time'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            out['tide_rms'] = float(np.sqrt(np.nanmean(tu ** 2 + tv ** 2)))
        u, v = inertial_notch(ru, rv)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        u, v = u - np.nanmean(u, axis=0), v - np.nanmean(v, axis=0)
    return u, v, out

# ───────────────────────── spectra with per-segment bookkeeping ─────────────────────────

def zonal_segments(u, v, L=SEG, max_fill=MAXFILL, max_fill_frac=0.15):
    """Gap-tolerant zonal segments of u (longitudinal) and v (transverse) with their hour index."""
    nt, ny, nx = u.shape
    A, C, H = [], [], []
    for t in range(nt):
        for j in range(ny):
            ra, rc = u[t, j], v[t, j]
            if np.isfinite(ra).sum() < L // 2:
                continue
            fa, ma = S.fill_short_gaps(ra, max_fill)
            fc, mc = S.fill_short_gaps(rc, max_fill)
            ok = np.isfinite(fa) & np.isfinite(fc)
            nfill = (ma | mc).astype(int)
            i = 0
            while i < nx:
                if not ok[i]:
                    i += 1; continue
                jj = i
                while jj < nx and ok[jj]:
                    jj += 1
                for s in range(i, jj - L + 1, L // 2):
                    if nfill[s:s + L].sum() <= max_fill_frac * L:
                        A.append(fa[s:s + L]); C.append(fc[s:s + L]); H.append(t)
                i = jj
    return np.asarray(A), np.asarray(C), np.asarray(H, int)


def psd(segs, dx_km):
    """Per-segment one-sided PSD over ALL bins (DC..Nyquist) and the untapered variance ⟨d²⟩."""
    L = segs.shape[1]
    d = segs - segs.mean(axis=1, keepdims=True)
    w = tukey(L, 0.5)
    X = np.fft.rfft(d * w, axis=1)
    P = np.abs(X) ** 2 * dx_km / (L * np.mean(w ** 2))
    P[:, 1:-1 if L % 2 == 0 else None] *= 2.0
    # Untapered check: the same estimator with w = 1 must integrate to ⟨d²⟩ exactly (DFT Parseval).
    Pb = np.abs(np.fft.rfft(d, axis=1)) ** 2 * dx_km / L
    Pb[:, 1:-1 if L % 2 == 0 else None] *= 2.0
    k = np.fft.rfftfreq(L, d=dx_km)
    box = Pb.sum(axis=1) * (k[1] - k[0])
    return k, P, (d ** 2).mean(axis=1), box


def month_blocks(path, kind):
    """Spectral sums per 7-day block for one month: dict block → (ΣP_L, ΣP_T, Σ⟨u_L²⟩, n)."""
    warnings.simplefilter('ignore', RuntimeWarning)
    meta = json.loads(str(np.load(path)['meta']))
    u, v, extra = filtered(path, kind)
    A, C, H = zonal_segments(u, v)
    if len(A) == 0:
        return meta['name'], None, extra
    k, PL, varL, boxL = psd(A, meta['dxKm'])
    _, PT, varT, boxT = psd(C, meta['dxKm'])
    blocks = {}
    for b in np.unique(H // BLOCK_H):
        m = (H // BLOCK_H) == b
        blocks[int(b)] = (PL[m].sum(0), PT[m].sum(0), varL[m].sum(), varT[m].sum(), int(m.sum()),
                          boxL[m].sum(), boxT[m].sum())
    return meta['name'], {'k': k, 'blocks': blocks}, extra

# ───────────────────────── group statistics ─────────────────────────

def group_stats(items, n_boot=2000, seed=0):
    """items: list of (k, blocks). Mean spectra, R, Parseval, and block-bootstrap 95 % CIs."""
    k = items[0][0]
    blocks = [b for _, bl in items for b in bl.values()]
    SL = np.array([b[0] for b in blocks]); ST = np.array([b[1] for b in blocks])
    VL = np.array([b[2] for b in blocks]); VT = np.array([b[3] for b in blocks]); N = np.array([b[4] for b in blocks])

    def est(idx):
        n = N[idx].sum()
        CL, CT = SL[idx].sum(0) / n, ST[idx].sum(0) / n
        return CL, CT, CT / CL
    CL, CT, R = est(np.arange(len(blocks)))
    rng = np.random.default_rng(seed)
    boot = [est(rng.integers(0, len(blocks), len(blocks))) for _ in range(n_boot)]
    ci = {key: np.percentile([b[i] for b in boot], [2.5, 97.5], axis=0) for i, key in enumerate(('CL', 'CT', 'R'))}
    dk = k[1] - k[0]
    BL = np.array([b[5] for b in blocks]); BT = np.array([b[6] for b in blocks])
    parseval = {'L': float(CL.sum() * dk / (VL.sum() / N.sum())), 'T': float(CT.sum() * dk / (VT.sum() / N.sum())),
                'L_untapered': float(BL.sum() / VL.sum()), 'T_untapered': float(BT.sum() / VT.sum())}
    return {'k': k, 'CL': CL, 'CT': CT, 'R': R, 'ci': ci, 'parseval': parseval, 'nseg': int(N.sum()),
            'nblocks': len(blocks)}


def f_branch(k_cpkm, coef):
    """QNSE f-branch coef·f²·k^-3 (k in rad/m) expressed per cycle/km."""
    k_rad = 2 * np.pi * k_cpkm / 1e3
    return coef * F ** 2 * k_rad ** -3 * (2 * np.pi / 1e3)


def kolmogorov_pi(k_cpkm, E_cpkm, branch_cpkm, factor):
    """Π (m²/s³) needed so that branch + factor·C_K·Π^(2/3)·k^-5/3 = E; NaN where E ≤ branch."""
    k_rad = 2 * np.pi * k_cpkm / 1e3
    excess = (E_cpkm - branch_cpkm) * 1e3 / (2 * np.pi)
    with np.errstate(invalid='ignore'):
        return np.where(excess > 0, (excess / (factor * C_K * k_rad ** (-5 / 3))) ** 1.5, np.nan)


def buhler_sensitivity(k, CL, CT):
    """Bühler split as computed, and with λ < 20 km replaced by a power law through the 26–52 km bins."""
    Dp0, Df0 = S.helmholtz_1d(k, CL, CT)
    wl = 1 / k
    fitb = (wl >= 25) & (wl <= 53)
    CLx, CTx = CL.copy(), CT.copy()
    slopes = {}
    for name, C, Cx in (('L', CL, CLx), ('T', CT, CTx)):
        p = np.polyfit(np.log(k[fitb]), np.log(C[fitb]), 1)
        slopes[name] = float(p[0])
        Cx[wl < FLOOR_KM] = np.exp(np.polyval(p, np.log(k[wl < FLOOR_KM])))
    Dp1, Df1 = S.helmholtz_1d(k, CLx, CTx)
    band = wl >= FLOOR_KM
    share = lambda a, b: float(a[band].sum() / (a[band] + b[band]).sum())
    return {'Dpsi': Dp0, 'Dphi': Df0, 'Dpsi_x': Dp1, 'Dphi_x': Df1, 'share': share(Dp0, Df0),
            'share_x': share(Dp1, Df1), 'extrap_slopes': slopes}

# ───────────────────────── validation ─────────────────────────

def expected_R(alpha, s):
    return (alpha * s + (1 - s)) / (s + alpha * (1 - s))


def validate_one(args):
    mask_path, alpha, share, seed = args
    warnings.simplefilter('ignore', RuntimeWarning)
    import tempfile
    real = np.load(mask_path)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 's.npz')
        V.synthetic_month(real, alpha, share, seed, p)
        name, res, _ = month_blocks(p, 'lanczos')
    if res is None:
        return None
    g = group_stats([(res['k'], res['blocks'])], n_boot=1)
    k, wl = g['k'][1:-1], 1 / g['k'][1:-1]
    CL, CT, R = g['CL'][1:-1], g['CT'][1:-1], g['R'][1:-1]
    band = wl >= FLOOR_KM
    sl = lambda E: float(np.polyfit(np.log(k[band]), np.log(E[band]), 1)[0])
    b = buhler_sensitivity(k, CL, CT)
    return {'alpha': alpha, 'share': share, 'slope_L': sl(CL), 'slope_T': sl(CT), 'R': float(np.mean(R[band])),
            'R_true': expected_R(alpha, share), 'share_est': b['share'], 'share_est_x': b['share_x'],
            'parseval_L': g['parseval']['L']}

# ───────────────────────── main ─────────────────────────

def main():
    warnings.simplefilter('ignore', RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz-dir', default=os.path.join(ROOT, 'analysis', 'npz'))
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'analysis', 'qnse_gistda'))
    ap.add_argument('--report-md', default=os.path.join(ROOT, 'analysis', 'report', 'qnse_gistda.md'))
    ap.add_argument('--realisations', type=int, default=50)
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    summ = json.load(open(os.path.join(ROOT, 'analysis', 'report', 'summary.json')))
    months = [m for m in summ['months'] if m['used']]
    paths = {m['ym']: os.path.join(a.npz_dir, f"hfr_{m['ym']}.npz") for m in months}
    meta = {m['ym']: m for m in months}

    # 1) spectra per month for both filters (parallel)
    res = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for kind in ('lanczos', 'utide_notch'):
            out = list(ex.map(month_blocks, [paths[y] for y in paths], [kind] * len(paths)))
            res[kind] = {y: o for y, o in zip(paths, out)}
    groups = {}
    for cfg in ('A', 'B'):
        for season in ('dry', 'wet'):
            ys = [y for y in paths if meta[y]['config'] == cfg and meta[y]['season'] == season]
            groups[f'{cfg}-{season}'] = ys
    stats = {kind: {} for kind in res}
    for kind in res:
        for gname, ys in groups.items():
            items = [(res[kind][y][1]['k'], res[kind][y][1]['blocks']) for y in ys if res[kind][y][1]]
            if items:
                stats[kind][gname] = group_stats(items)
    tide_rms = {y: res['utide_notch'][y][2].get('tide_rms') for y in paths}

    # 2) validation ensemble on best / median / worst months (by zonal segment count)
    nseg = {y: sum(b[4] for b in res['lanczos'][y][1]['blocks'].values()) if res['lanczos'][y][1] else 0 for y in paths}
    order = sorted(paths, key=lambda y: nseg[y])
    masks = {'worst': order[0], 'median': order[len(order) // 2], 'best': order[-1]}
    jobs = [(paths[y], alpha, share, seed) for y in masks.values() for alpha in (5 / 3, 3.0) for share in (1.0, 0.5)
            for seed in range(a.realisations)]
    with ProcessPoolExecutor(a.jobs) as ex:
        vres = list(ex.map(validate_one, jobs, chunksize=4))
    val = {}
    for (pth, alpha, share, seed), r in zip(jobs, vres):
        if r is None:
            continue
        mname = next(k for k, y in masks.items() if paths[y] == pth)
        val.setdefault((mname, round(alpha, 2), share), []).append(r)

    # 3) write figures, CSVs, markdown
    write_outputs(a, stats, groups, tide_rms, masks, nseg, val)


def fmt_ci(x, lo, hi, f='.3g'):
    return f'{x:{f}} [{lo:{f}}, {hi:{f}}]'


def write_outputs(a, stats, groups, tide_rms, masks, nseg, val):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    kinds = {'utide_notch': '(a) UTide + inertial notch', 'lanczos': '(b) Lanczos 72 h'}
    gnames = [g for g in ('A-dry', 'A-wet', 'B-dry', 'B-wet') if g in stats['lanczos']]

    # spectra + f-branch, per group
    fig, axes = plt.subplots(2, len(gnames), figsize=(4.4 * len(gnames), 8), squeeze=False)
    for col, g in enumerate(gnames):
        for row, comp, coef in ((0, 'CL', QL), (1, 'CT', QT)):
            ax = axes[row][col]
            for kind, color in (('utide_notch', '#d62728'), ('lanczos', '#1f77b4')):
                s = stats[kind].get(g)
                if not s:
                    continue
                k = s['k'][1:-1]
                ax.fill_between(k, s['ci'][comp][0][1:-1], s['ci'][comp][1][1:-1], color=color, alpha=0.2, lw=0)
                ax.loglog(k, s[comp][1:-1], 'o-', ms=3, color=color, label=kinds[kind])
            kk = stats['lanczos'][g]['k'][1:-1]
            ax.loglog(kk, f_branch(kk, coef), 'k--', lw=1.4, label=f'QNSE f-branch {coef} f²k⁻³')
            ax.axvspan(1 / FLOOR_KM, kk[-1] * 1.05, color='0.85', alpha=0.5, lw=0)
            ax.set_title(f"{g}: {'C_L' if comp == 'CL' else 'C_T'} (zonal, n = {len(groups[g])} mo)", fontsize=9)
            ax.set_xlabel('k [cycles/km]')
            ax.grid(True, which='both', alpha=0.25)
            if col == 0:
                ax.set_ylabel('(m/s)² per cycle/km')
                ax.legend(fontsize=7)
            sec = ax.secondary_xaxis('top', functions=(lambda x: 2 * np.pi * x / 1e3, lambda x: x * 1e3 / (2 * np.pi)))
            sec.set_xlabel('k [rad/m]', fontsize=7)
    fig.suptitle(f'GISTDA zonal spectra vs the parameter-free QNSE f-branch (f at {LAT}°N); bands: 95 % block bootstrap',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'spectra_vs_fbranch.png'), dpi=130)
    plt.close(fig)

    # R = C_T / C_L
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    for g, c in zip(gnames, ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728')):
        s = stats['lanczos'][g]
        k = s['k'][1:-1]
        ax.fill_between(k, s['ci']['R'][0][1:-1], s['ci']['R'][1][1:-1], color=c, alpha=0.15, lw=0)
        ax.semilogx(k, s['R'][1:-1], 'o-', ms=3, color=c, label=g)
    for y, lab in ((QT / QL, 'QNSE f-branch 2.59'), (5 / 3, 'rotational k⁻⁵/³: 1.67'), (3, 'rotational k⁻³: 3'), (1, 'isotropic noise: 1')):
        ax.axhline(y, ls=':', lw=1, color='0.4'); ax.text(k[0], y * 1.02, lab, fontsize=7)
    ax.axvspan(1 / FLOOR_KM, k[-1] * 1.05, color='0.85', alpha=0.5, lw=0)
    ax.set_xlabel('k [cycles/km]'); ax.set_ylabel('R = C_T / C_L'); ax.set_title('R per bin, (b) Lanczos 72 h, 95 % CI', fontsize=10)
    ax.grid(True, which='both', alpha=0.25); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'R_ratio.png'), dpi=130)
    plt.close(fig)

    # CSV in both units
    with open(os.path.join(a.out_dir, 'spectra_zonal_by_group.csv'), 'w') as fh:
        fh.write('filter,group,k_cpkm,k_radm,wavelength_km,C_L_cpkm,C_L_lo,C_L_hi,C_T_cpkm,C_T_lo,C_T_hi,'
                 'C_L_radm,C_T_radm,R,R_lo,R_hi,fbranch_L_cpkm,fbranch_T_cpkm,obs_over_QNSE_L,obs_over_QNSE_T\n')
        for kind in stats:
            for g, s in stats[kind].items():
                for i in range(1, len(s['k']) - 1):
                    k = s['k'][i]
                    bL, bT = f_branch(k, QL), f_branch(k, QT)
                    fh.write(f"{kind},{g},{k:.6g},{2 * np.pi * k / 1e3:.6g},{1 / k:.4g},{s['CL'][i]:.6g},{s['ci']['CL'][0][i]:.6g},"
                             f"{s['ci']['CL'][1][i]:.6g},{s['CT'][i]:.6g},{s['ci']['CT'][0][i]:.6g},{s['ci']['CT'][1][i]:.6g},"
                             f"{s['CL'][i] * 1e3 / (2 * np.pi):.6g},{s['CT'][i] * 1e3 / (2 * np.pi):.6g},{s['R'][i]:.4g},"
                             f"{s['ci']['R'][0][i]:.4g},{s['ci']['R'][1][i]:.4g},{bL:.6g},{bT:.6g},{s['CL'][i] / bL:.4g},{s['CT'][i] / bT:.4g}\n")

    # ── markdown ──
    L = ['## QNSE comparison for the Gulf of Thailand (13.1°N) — follow-up analysis\n',
         '_Generated by `tools/hfr_qnse_gistda.py`. Figures and CSV in `analysis/qnse_gistda/`._\n']
    L += ['### 1. Spectral convention and Parseval check\n',
          '* One-sided wavenumber spectra; k in cycles/km (rad/m given alongside: k_rad = 2πk/1000), E in (m/s)² per',
          '  cycle/km (rad/m: E_rad = E·1000/2π, units m³/s²).',
          '* Each 52 km zonal segment is de-meaned (not detrended), Tukey(0.5)-tapered, transformed, scaled by',
          '  Δx/(L⟨w²⟩) and doubled except DC and Nyquist, so ∫E dk = ⟨(d·w)²⟩/⟨w²⟩: the taper loss is corrected.',
          '* Parseval check, ∫C dk / ⟨u²⟩ over all bins (DC to Nyquist), per group:\n',
          '| Filter | Group | Segments | 7-day blocks | untapered ∫C_L dk/⟨u_L²⟩ | untapered T | tapered ∫C_L dk/⟨u_L²⟩ | tapered T |',
          '|---|---|---|---|---|---|---|---|']
    for kind in stats:
        for g, s in stats[kind].items():
            p = s['parseval']
            L.append(f"| {kinds[kind]} | {g} | {s['nseg']:,} | {s['nblocks']} | {p['L_untapered']:.4f} | {p['T_untapered']:.4f} | "
                     f"{p['L']:.3f} | {p['T']:.3f} |")
    L += ['\n* **Untapered = 1.0000** verifies the one-sided normalisation exactly (DFT Parseval).',
          '* **Tapered < 1** (real data 0.53–1.06; homogeneous synthetic fields 0.61–0.83 in section 7, lower for k⁻³ than',
          '  for k⁻⁵/³). The ⟨w²⟩ correction is exact only for variance spread evenly along the segment. Energy at scales',
          '  longer than the 52 km segment appears as a gradient across it, concentrated at the segment ends where the',
          '  Tukey taper down-weights it, so that part of ⟨u²⟩ is not recovered. The synthetic ensemble includes this',
          '  effect, so the validated biases in section 7 already account for it. Spectra are not rescaled.\n']

    L += ['### 2. Parameter-free QNSE f-branch versus observations (zonal, per season and configuration)\n',
          f'f = 2Ω sin {LAT}° = {F:.3e} s⁻¹. f-branch: 0.0926 f²k⁻³ (C_L) and 0.24 f²k⁻³ (C_T), k in rad/m, converted',
          'to cycles/km. "obs/QNSE" is the observed spectrum divided by the f-branch in each bin (λ ≥ 20 km in bold).\n']
    for kind in ('utide_notch', 'lanczos'):
        L += [f'**{kinds[kind]}**\n', '| Group | λ [km] | C_L (95 % CI) | obs/QNSE (L) | C_T (95 % CI) | obs/QNSE (T) | R = C_T/C_L (95 % CI) |',
              '|---|---|---|---|---|---|---|']
        verdict = []
        for g in gnames:
            s = stats[kind].get(g)
            if not s:
                continue
            below = False
            for i in range(1, len(s['k']) - 1):
                k = s['k'][i]; wl = 1 / k
                if wl < 9:
                    continue
                bL, bT = f_branch(k, QL), f_branch(k, QT)
                rl, rt = s['CL'][i] / bL, s['CT'][i] / bT
                strong = wl >= FLOOR_KM
                if strong and (rl < 1 or rt < 1):
                    below = True
                b = (lambda x: f'**{x}**') if strong else (lambda x: x)
                L.append(f"| {g} | {b(f'{wl:.1f}')} | {fmt_ci(s['CL'][i], *s['ci']['CL'][:, i])} | {b(f'{rl:.3f}')} | "
                         f"{fmt_ci(s['CT'][i], *s['ci']['CT'][:, i])} | {b(f'{rt:.3f}')} | {fmt_ci(s['R'][i], *s['ci']['R'][:, i], '.2f')} |")
            verdict.append((g, below))
        L.append('')
        for g, below in verdict:
            if below:
                L.append(f'* {kinds[kind]}, {g}: the observed spectrum lies **below** the QNSE f-branch in the fitted band. '
                         'Because QNSE adds a non-negative Kolmogorov term to the f-branch, **no Π ≥ 0 can fit**; no fit is attempted.')
            else:
                s = stats[kind][g]
                band = (1 / s['k'][1:-1]) >= FLOOR_KM
                pis = kolmogorov_pi(s['k'][1:-1][band], s['CL'][1:-1][band], f_branch(s['k'][1:-1][band], QL), 18 / 55)
                L.append(f'* {kinds[kind]}, {g}: observations exceed the f-branch in the fitted band; the Π implied bin by bin '
                         f'(C_K = 1.6, E_L) is ' + ', '.join(f'{p:.2e}' for p in pis) + ' m²/s³ — reported, not fitted.')
        L.append('')

    L += ['### 3. Primary diagnostics: C_L, C_T and R per bin\n',
          'The tables above give C_L, C_T and R = C_T/C_L with 95 % intervals from a bootstrap over 7-day blocks',
          '(subtidal fields are correlated for several days, so single days are not independent). Reference values of',
          'R: 2.59 for the QNSE f-branch (0.24/0.0926), 1.67 and 3 for isotropic non-divergent k⁻⁵/³ and k⁻³ spectra,',
          '1 for isotropic white noise. Figure: `analysis/qnse_gistda/R_ratio.png`. The Bühler split is now secondary.\n',
          '![Spectra vs f-branch](../qnse_gistda/spectra_vs_fbranch.png)\n', '![R](../qnse_gistda/R_ratio.png)\n']

    L += ['### 4. Like-with-like: zonal-only spectra, configurations A and B separately\n',
          'All spectra in sections 2–3 are zonal-only and grouped by configuration (A: to Oct 2025, B: from Nov 2025)',
          'and season, so each comparison uses one footprint and one transect direction.\n']

    L += ['### 5. Time-filter sensitivity\n', '| Group | λ [km] | C_L (a)/(b) | C_T (a)/(b) |', '|---|---|---|---|']
    for g in gnames:
        sa, sb = stats['utide_notch'].get(g), stats['lanczos'].get(g)
        if not (sa and sb):
            continue
        for i in range(1, len(sb['k']) - 1):
            wl = 1 / sb['k'][i]
            if wl >= 9:
                L.append(f"| {g} | {wl:.1f} | {sa['CL'][i] / sb['CL'][i]:.2f} | {sa['CT'][i] / sb['CT'][i]:.2f} |")
    L += ['\n(a) keeps all non-tidal motion except the clockwise inertial band (UTide, automatic constituent',
          f'selection with Rayleigh criterion 1, per node; notch = complex demodulation at −f, inertial period',
          f'{T_INERTIAL_H:.1f} h, 120 h envelope low-pass); (b) keeps only periods > 72 h.\n']

    L += ['### 6. Bühler split sensitivity to the unresolved small scales\n',
          '| Filter | Group | Rotational share, λ ≥ 20 km (as computed) | with λ < 20 km replaced by a power law through 26–52 km | Extrapolation slopes C_L / C_T |',
          '|---|---|---|---|---|']
    for kind in stats:
        for g, s in stats[kind].items():
            b = buhler_sensitivity(s['k'][1:-1], s['CL'][1:-1], s['CT'][1:-1])
            L.append(f"| {kinds[kind]} | {g} | {b['share']:.2f} | {b['share_x']:.2f} | {b['extrap_slopes']['L']:+.2f} / {b['extrap_slopes']['T']:+.2f} |")
    L.append('\nA share outside 0–1 indicates that the isotropy assumption of the split fails for that group.\n')

    L += ['### 7. Validation (synthetic months, real gap masks)\n',
          f"{a.realisations} realisations per case; masks: best {masks['best']} ({nseg[masks['best']]:,} segments), "
          f"median {masks['median']} ({nseg[masks['median']]:,}), worst {masks['worst']} ({nseg[masks['worst']]:,}).",
          'Synthetic fields carry energy from 300 km down to the grid scale (so at scales > 52 km), with K1 tide,',
          '53 h inertial oscillation and 3 cm/s noise, and pass through the exact (b) pipeline.\n',
          '| Mask | True slope | True rot. share | Slope C_L bias ± sd | Slope C_T bias ± sd | R (true) | R bias ± sd | Bühler share bias ± sd | with extrapolation | Parseval L |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for (mname, alpha, share), rs in sorted(val.items()):
        sl_L = np.array([r['slope_L'] for r in rs]) + alpha
        sl_T = np.array([r['slope_T'] for r in rs]) + alpha
        Rb = np.array([r['R'] - r['R_true'] for r in rs])
        sb = np.array([r['share_est'] - share for r in rs]); sx = np.array([r['share_est_x'] - share for r in rs])
        L.append(f"| {mname} | −{alpha:.2f} | {share:.1f} | {sl_L.mean():+.2f} ± {sl_L.std():.2f} | {sl_T.mean():+.2f} ± {sl_T.std():.2f} | "
                 f"{rs[0]['R_true']:.2f} | {Rb.mean():+.2f} ± {Rb.std():.2f} | {sb.mean():+.2f} ± {sb.std():.2f} | "
                 f"{sx.mean():+.2f} ± {sx.std():.2f} | {np.mean([r['parseval_L'] for r in rs]):.3f} |")
    L.append('')

    L += ['### 8. Bottom friction versus rotation\n',
          f'Tidal current RMS speed from the UTide reconstruction (per month, all nodes). Quadratic drag C_d = {CD:.1e};',
          'linear rate r = C_d U_tide / H (Lorentz linearisation would multiply by 8/(3π) ≈ 0.85). Depths: upper Gulf of',
          'Thailand / Bight of Bangkok, H = 10, 15 (typical) and 25 m.\n',
          '| Month | Season | Config | U_tide RMS [m/s] | r [s⁻¹], H = 10 / 15 / 25 m | r/f, H = 10 / 15 / 25 m |', '|---|---|---|---|---|---|']
    summ = json.load(open(os.path.join(ROOT, 'analysis', 'report', 'summary.json')))
    mm = {m['ym']: m for m in summ['months']}
    allU = []
    for y, U in sorted(tide_rms.items()):
        if U is None:
            continue
        allU.append(U)
        rs = [CD * U / h for h in H_DEPTHS]
        L.append(f"| {y[:4]}-{y[4:]} | {mm[y]['season']} | {mm[y]['config']} | {U:.3f} | "
                 + ' / '.join(f'{r:.1e}' for r in rs) + ' | ' + ' / '.join(f'{r / F:.2f}' for r in rs) + ' |')
    Um = float(np.mean(allU))
    L.append(f"\nMean U_tide = {Um:.3f} m/s ⇒ r = {CD * Um / 15:.1e} s⁻¹ at H = 15 m, i.e. r/f = {CD * Um / 15 / F:.2f} "
             f"(range {CD * Um / 25 / F:.2f}–{CD * Um / 10 / F:.2f} for H = 25–10 m).\n")
    with open(a.report_md, 'w') as fh:
        fh.write('\n'.join(L))
    json.dump({'tide_rms': tide_rms, 'masks': masks, 'f': F, 'U_mean': Um},
              open(os.path.join(a.out_dir, 'summary.json'), 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
