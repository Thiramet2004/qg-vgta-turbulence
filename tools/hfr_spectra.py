#!/usr/bin/env python3
"""hfr_spectra.py — subtidal 1-D kinetic-energy spectra of gridded HF radar currents, with a
rotational/divergent (Helmholtz) decomposition and scale-masked power-law fits.

Pipeline (each step switchable):
 1. Time filter — default a Lanczos low-pass, 72 h half-power cutoff, ±120 h window. It removes
    the tides and the near-inertial band: at 13°N the inertial period 2π/f ≈ 53 h lies *below*
    the ~30–40 h cutoff of the Godin filter, so Godin leaves it in. Responses are printed.
 2. Spectra — 1-D longitudinal C_L and transverse C_T along zonal and meridional transects from
    gap-tolerant segments: interior gaps of ≤ --max-fill nodes are linearly filled (≤ 15 % of a
    segment), then segments are de-meaned, Tukey(0.5)-tapered and averaged (Welch, 50 % overlap).
 3. Helmholtz decomposition of the 1-D spectra (Bühler, Callies & Ferrari 2014, J. Fluid Mech.
    757, 255–271), assuming horizontal isotropy:
        D_ψ(k) = C_T(k) + (1/k) ∫_k^∞ [C_T(κ) − C_L(κ)] dκ      (rotational)
        D_φ(k) = C_L(k) − (1/k) ∫_k^∞ [C_T(κ) − C_L(κ)] dκ      (divergent)
    D_ψ + D_φ = C_L + C_T. Isotropic white noise (C_L = C_T) cancels in the integral.
 4. Fits — E ∝ k^-α fitted only for wavelengths ≥ --min-wavelength-km (default 20 km): shorter
    scales are steepened by the radar's least-squares combination of radials into totals.

Import `analyze()` for batch use (tools/hfr_report.py).

Example:
  python tools/hfr_spectra.py analysis/npz/hfr_202605.npz --segment 32 --plot spectra.png
"""
import argparse
import json
import math
import warnings

import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal.windows import tukey

R_KM = 6371.0

# ───────────────────────── time filtering ─────────────────────────

def lanczos_weights(cutoff_h, half_len_h, dt_h=1.0):
    n = int(round(half_len_h / dt_h))
    k = np.arange(-n, n + 1)
    fc = dt_h / cutoff_h
    w = 2 * fc * np.sinc(2 * fc * k) * np.sinc(k / (n + 1))
    return w / w.sum()


def filter_gain(w, period_h, dt_h=1.0):
    n = (len(w) - 1) // 2
    k = np.arange(-n, n + 1)
    return float(np.sum(w * np.cos(2 * np.pi * k * dt_h / period_h)))


def lowpass(a, w, max_missing=0.2):
    """NaN-aware convolution along time (axis 0). Output is NaN where more than `max_missing`
    of the window's |weight| falls on missing samples or outside the record."""
    ok = np.isfinite(a)
    x = np.where(ok, a, 0.0)
    num = convolve1d(x, w, axis=0, mode='constant', cval=0.0)
    den = convolve1d(ok.astype(float), w, axis=0, mode='constant', cval=0.0)
    miss = 1.0 - convolve1d(ok.astype(float), np.abs(w), axis=0, mode='constant', cval=0.0) / np.abs(w).sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / np.where(np.abs(den) > 1e-6, den, np.nan)
    out[miss > max_missing] = np.nan
    return out


def running_mean(a, n, min_frac=0.8):
    ok = np.isfinite(a)
    w = np.ones(n) / n
    num = convolve1d(np.where(ok, a, 0.0), w, axis=0, mode='constant')
    cnt = convolve1d(ok.astype(float), w, axis=0, mode='constant')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / np.where(cnt > 0, cnt, np.nan)
    out[cnt < min_frac] = np.nan
    return out


# Main tidal constituents: periods in hours.
TIDES_H = {'M2': 12.4206012, 'S2': 12.0, 'N2': 12.65834751, 'K2': 11.96723606, 'K1': 23.93447213,
           'O1': 25.81933871, 'P1': 24.06588766, 'Q1': 26.868350, 'M4': 6.210300601, 'MS4': 6.103339275, 'M6': 4.140200401}


def detide_harmonic(a, step_h=1.0, min_frac=0.6):
    """Remove only the deterministic tides: least-squares fit of the TIDES_H constituents (plus a
    mean) at every grid node over the record; returns the residual (all non-tidal motion kept,
    including near-inertial and wind-driven currents). Nodes with < min_frac valid hours → NaN."""
    nt = a.shape[0]
    t = np.arange(nt) * step_h
    w = 2 * np.pi / np.array(list(TIDES_H.values()))
    X = np.column_stack([np.ones(nt)] + [f(np.outer(t, w)) for f in (np.cos, np.sin)]).reshape(nt, -1)
    flat = a.reshape(nt, -1)
    out = np.full_like(flat, np.nan)
    ok = np.isfinite(flat)
    for j in np.where(ok.mean(axis=0) >= min_frac)[0]:
        m = ok[:, j]
        coef, *_ = np.linalg.lstsq(X[m], flat[m, j], rcond=None)
        out[m, j] = flat[m, j] - X[m] @ coef + coef[0]           # keep the mean (removed later if anomaly)
    return out.reshape(a.shape)


def godin(a):
    """Godin A24·A24·A25 (kept for comparison; does NOT remove the 53 h inertial band at 13°N)."""
    return running_mean(running_mean(running_mean(a, 24), 24), 25)

# ───────────────────────── spatial spectra ─────────────────────────

def fill_short_gaps(row, max_fill):
    """Linearly fill interior NaN runs of length ≤ max_fill. Returns (filled row, filled mask)."""
    out, filled = row.copy(), np.zeros(len(row), bool)
    if max_fill <= 0:
        return out, filled
    ok = np.isfinite(row)
    i, n = 0, len(row)
    while i < n:
        if ok[i]:
            i += 1
            continue
        j = i
        while j < n and not ok[j]:
            j += 1
        if 0 < i and j < n and j - i <= max_fill:                 # interior gap with data both sides
            t = (np.arange(i, j) - (i - 1)) / (j - (i - 1))
            out[i:j] = row[i - 1] + t * (row[j] - row[i - 1])
            filled[i:j] = True
        i = j
    return out, filled


def segments(along, across, axis, L, max_fill, max_fill_frac=0.15):
    """Matching gap-tolerant length-L segments of the along- and across-transect components."""
    A = np.moveaxis(along, axis, -1).reshape(-1, along.shape[axis])
    C = np.moveaxis(across, axis, -1).reshape(-1, across.shape[axis])
    sa, sc = [], []
    for ra, rc in zip(A, C):
        if np.isfinite(ra).sum() < L // 2:
            continue
        fa, ma = fill_short_gaps(ra, max_fill)
        fc, mc = fill_short_gaps(rc, max_fill)
        ok = np.isfinite(fa) & np.isfinite(fc)
        nfill = (ma | mc).astype(int)
        i, n = 0, len(ra)
        while i < n:
            if not ok[i]:
                i += 1
                continue
            j = i
            while j < n and ok[j]:
                j += 1
            for s in range(i, j - L + 1, max(1, L // 2)):           # 50 % overlap
                if nfill[s:s + L].sum() <= max_fill_frac * L:
                    sa.append(fa[s:s + L]); sc.append(fc[s:s + L])
            i = j
    return np.asarray(sa), np.asarray(sc)


PREWHITEN = False      # optional: first-difference before tapering, recolour after (did not reduce bias in validation)


def periodogram(segs, dx_km, prewhiten=None):
    """Mean one-sided PSD [(m/s)² per cycle/km]; drops DC and the Nyquist bin.
    Mean removal + Tukey(0.5) taper: linear detrending and a full Hann taper bias the lowest
    bins unequally for the along- and across-transect components (verified on synthetic fields),
    which the 1/k factor of the Helmholtz decomposition then amplifies."""
    prewhiten = PREWHITEN if prewhiten is None else prewhiten
    L = segs.shape[1]
    d = segs - segs.mean(axis=1, keepdims=True)
    if prewhiten:
        # Differencing turns a steep k^-n spectrum into a nearly flat one, so the taper's sidelobes no
        # longer carry large-scale energy into high wavenumbers; the |1 − e^{-ikΔx}|² response is
        # divided out afterwards. Padding keeps the length (and the k grid) unchanged.
        d = np.diff(d, axis=1, prepend=d[:, :1])
    w = tukey(L, 0.5)
    X = np.fft.rfft(d * w, axis=1)
    P = np.abs(X) ** 2 * dx_km / (L * np.mean(w ** 2))
    P[:, 1:-1 if L % 2 == 0 else None] *= 2.0
    k = np.fft.rfftfreq(L, d=dx_km)
    end = -1 if L % 2 == 0 else None
    k, P = k[1:end], P[:, 1:end].mean(axis=0)
    if prewhiten:
        P = P / (4 * np.sin(np.pi * k * dx_km) ** 2)
    return k, P


def helmholtz_1d(k, CL, CT):
    """Bühler, Callies & Ferrari (2014) decomposition of 1-D spectra into rotational D_ψ and
    divergent D_φ parts. The tail integral ∫_k^∞ is truncated at the highest resolved bin
    (trapezoid on the uniform k grid); with C_T ≈ C_L in the noise plateau the truncation is small."""
    g = CT - CL
    dk = k[1] - k[0] if len(k) > 1 else 1.0
    tail = np.cumsum(g[::-1])[::-1] * dk - 0.5 * g * dk         # ∫_k^kmax g dκ
    return CT + tail / k, CL - tail / k


def fit_power_law(k, E, min_wl_km, max_wl_km=None):
    """Least-squares fit E ∝ k^slope over wavelengths [min_wl, max_wl]. Returns (slope, stderr, n)."""
    wl = 1.0 / k
    sel = (wl >= min_wl_km) & (E > 0) & np.isfinite(E)
    if max_wl_km:
        sel &= wl <= max_wl_km
    n = int(sel.sum())
    if n < 2:
        return math.nan, math.nan, n
    x, y = np.log(k[sel]), np.log(E[sel])
    p = np.polyfit(x, y, 1)
    if n < 3:
        return p[0], math.nan, n
    res = y - np.polyval(p, x)
    se = math.sqrt(np.sum(res ** 2) / (n - 2) / np.sum((x - x.mean()) ** 2))
    return p[0], se, n

# ───────────────────────── analysis driver ─────────────────────────

def analyze(path, lowpass_kind='lanczos', cutoff_h=72.0, half_len_h=120.0, anomaly=True,
            segment=32, max_fill=2, min_wl_km=20.0, max_wl_km=None, segment_meridional=None):
    """Full analysis of one gridded month. Returns a dict (all spectra in (m/s)² per cycle/km).
    `segment_meridional` sets a separate (usually shorter) segment length for meridional transects,
    because the footprint is narrower in that direction."""
    d = np.load(path)
    meta = json.loads(str(d['meta']))
    u, v = d['u'].astype(np.float64), d['v'].astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)          # never-observed cells
        raw_var = np.nanvar(u) + np.nanvar(v)
    info = {'name': meta['name'], 'meta': meta, 'filter': lowpass_kind,
            'raw_hours': int(np.any(np.isfinite(u), axis=(1, 2)).sum())}
    if lowpass_kind == 'lanczos':
        w = lanczos_weights(cutoff_h, half_len_h, meta['stepHours'])
        info['gain'] = {T: filter_gain(w, T, meta['stepHours']) for T in (12.42, 24.0, 53.3, 72.0, 120.0)}
        u, v = lowpass(u, w), lowpass(v, w)
    elif lowpass_kind == 'godin':
        u, v = godin(u), godin(v)
    elif lowpass_kind == 'harmonic':
        u, v = detide_harmonic(u, meta['stepHours']), detide_harmonic(v, meta['stepHours'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        info['subtidal_frac'] = (np.nanvar(u) + np.nanvar(v)) / raw_var
        info['valid_hours'] = int(np.any(np.isfinite(u), axis=(1, 2)).sum())
        if anomaly:
            u, v = u - np.nanmean(u, axis=0), v - np.nanmean(v, axis=0)

    if meta.get('dxKm'):
        dx = dy = meta['dxKm']
    else:
        dy = math.radians(meta['dlat']) * R_KM
        dx = math.radians(meta['dlon']) * R_KM * math.cos(math.radians(float(np.mean(d['lat']))))
    out = {}
    seg_len = {'zonal': segment, 'meridional': segment_meridional or segment}
    for name, axis, dxk, along, across in (('zonal', 2, dx, u, v), ('meridional', 1, dy, v, u)):
        sL, sT = segments(along, across, axis, seg_len[name], max_fill)
        if len(sL) < 10:
            out[name] = None
            continue
        k, CL = periodogram(sL, dxk)
        _, CT = periodogram(sT, dxk)
        Dpsi, Dphi = helmholtz_1d(k, CL, CT)
        out[name] = {'k': k, 'CL': CL, 'CT': CT, 'Dpsi': Dpsi, 'Dphi': Dphi, 'nseg': len(sL), 'dx': dxk}
    # Isotropic estimate: segment-weighted average of the two directions on the same k grid.
    z, m = out.get('zonal'), out.get('meridional')
    if z and m and z['k'].shape == m['k'].shape and np.allclose(z['k'], m['k']):
        nz, nm = z['nseg'], m['nseg']
        out['combined'] = {key: (z[key] * nz + m[key] * nm) / (nz + nm) for key in ('CL', 'CT', 'Dpsi', 'Dphi')}
        out['combined'].update(k=z['k'], nseg=nz + nm, dx=z['dx'])
    elif z and m and abs(z['dx'] - m['dx']) / m['dx'] < 0.35:
        # Different spacings (lat/lon grids): put the meridional spectra on the zonal k grid (log-log
        # interpolation inside the common range) and average segment-weighted where both exist.
        nz, nm = z['nseg'], m['nseg']
        comb = {}
        for key in ('CL', 'CT', 'Dpsi', 'Dphi'):
            mi = np.interp(np.log(z['k']), np.log(m['k']), m[key], left=np.nan, right=np.nan)
            comb[key] = np.where(np.isfinite(mi), (z[key] * nz + mi * nm) / (nz + nm), z[key])
        out['combined'] = dict(comb, k=z['k'], nseg=nz + nm, dx=z['dx'])
    else:
        out['combined'] = z or m
    for s in out.values():
        if not s:
            continue
        s['fit'] = {key: fit_power_law(s['k'], s[key], min_wl_km, max_wl_km) for key in ('CL', 'CT', 'Dpsi', 'Dphi')}
        band = (1.0 / s['k']) >= min_wl_km
        tot = np.sum(s['Dpsi'][band] + s['Dphi'][band])
        s['rot_frac'] = float(np.sum(s['Dpsi'][band]) / tot) if tot > 0 else math.nan
    info['spectra'] = out
    info['settings'] = dict(segment=segment, segment_meridional=seg_len['meridional'], max_fill=max_fill, min_wl_km=min_wl_km, max_wl_km=max_wl_km,
                            cutoff_h=cutoff_h, half_len_h=half_len_h, anomaly=anomaly)
    return info


def structure_functions(u, v, dx, dy, max_lag):
    """Second-order longitudinal/transverse structure functions along x and y (all valid pairs)."""
    out = {}
    for name, axis, d, along, across in (('zonal', 2, dx, u, v), ('meridional', 1, dy, v, u)):
        r, DL, DT = [], [], []
        for lag in range(1, max_lag + 1):
            a0, a1 = [slice(None)] * 3, [slice(None)] * 3
            a0[axis], a1[axis] = slice(lag, None), slice(None, -lag)
            dL = along[tuple(a0)] - along[tuple(a1)]
            dT = across[tuple(a0)] - across[tuple(a1)]
            ok = np.isfinite(dL) & np.isfinite(dT)
            if ok.sum() < 200:
                break
            r.append(lag * d); DL.append(np.mean(dL[ok] ** 2)); DT.append(np.mean(dT[ok] ** 2))
        out[name] = tuple(np.asarray(x) for x in (r, DL, DT))
    return out

# ───────────────────────── CLI ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz', help='output of hfr_preprocess.py')
    ap.add_argument('--lowpass', choices=['lanczos', 'godin', 'harmonic', 'none'], default='lanczos',
                    help='lanczos: subtidal (> 72 h); harmonic: remove tidal constituents only; none: raw')
    ap.add_argument('--cutoff-hours', type=float, default=72.0, help='Lanczos half-power period (default 72 h)')
    ap.add_argument('--half-length-hours', type=float, default=120.0, help='Lanczos half-window (default 120 h)')
    ap.add_argument('--no-anomaly', action='store_true', help='keep the time mean at each grid point')
    ap.add_argument('--segment', type=int, default=32, help='segment length in grid points (default 32)')
    ap.add_argument('--segment-meridional', type=int, default=24,
                    help='segment length for meridional transects (default 24: the footprint is narrower N–S)')
    ap.add_argument('--max-fill', type=int, default=2, help='fill interior gaps up to this many nodes (default 2)')
    ap.add_argument('--min-wavelength-km', type=float, default=20.0,
                    help='exclude shorter wavelengths from fits (radar smoothing; default 20 km)')
    ap.add_argument('--max-wavelength-km', type=float, default=None)
    ap.add_argument('--plot', help='write a figure (PNG) — needs matplotlib')
    ap.add_argument('--csv', help='write k and all spectra as CSV')
    a = ap.parse_args()

    r = analyze(a.npz, a.lowpass, a.cutoff_hours, a.half_length_hours, not a.no_anomaly,
                a.segment, a.max_fill, a.min_wavelength_km, a.max_wavelength_km, a.segment_meridional)
    print(f"{r['name']}: filter {r['filter']}", end='')
    if 'gain' in r:
        print('  (gain ' + ', '.join(f'{T:g} h {g:+.3f}' for T, g in r['gain'].items()) + ')', end='')
    print(f"\n  subtidal share of KE variance {r['subtidal_frac']:.1%}; hours with data {r['raw_hours']} raw, "
          f"{r['valid_hours']} after filtering")
    fmt = lambda t: f'{t[0]:+.2f}±{t[1]:.2f}' if np.isfinite(t[1]) else f'{t[0]:+.2f} (n={t[2]})'
    for name in ('zonal', 'meridional', 'combined'):
        s = r['spectra'].get(name)
        if not s:
            print(f'  {name:10s} too few segments (try a smaller --segment or larger --max-fill)')
            continue
        f = s['fit']
        print(f"  {name:10s} segs {s['nseg']:6d}  λ≥{a.min_wavelength_km:g} km ({f['Dpsi'][2]} bins):"
              f"  C_L {fmt(f['CL'])}  C_T {fmt(f['CT'])}  D_ψ {fmt(f['Dpsi'])}  D_φ {fmt(f['Dphi'])}"
              f"  rotational share {s['rot_frac']:.0%}")

    if a.csv:
        with open(a.csv, 'w') as fh:
            fh.write('direction,k_cpkm,wavelength_km,C_L,C_T,D_psi,D_phi\n')
            for name, s in r['spectra'].items():
                if s:
                    for i, k in enumerate(s['k']):
                        fh.write(f"{name},{k:.6g},{1 / k:.4g},{s['CL'][i]:.6g},{s['CT'][i]:.6g},"
                                 f"{s['Dpsi'][i]:.6g},{s['Dphi'][i]:.6g}\n")
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names = [n for n in ('zonal', 'meridional', 'combined') if r['spectra'].get(n)]
        fig, axes = plt.subplots(2, len(names), figsize=(4.8 * len(names), 8), squeeze=False)
        labels = {'CL': '$C_L$', 'CT': '$C_T$', 'Dpsi': r'$D_\psi$ (rotational)', 'Dphi': r'$D_\phi$ (divergent)'}
        for col, name in enumerate(names):
            s = r['spectra'][name]
            k = s['k']
            for row, keys in enumerate((('CL', 'CT'), ('Dpsi', 'Dphi'))):
                ax = axes[row][col]
                for key, mk in zip(keys, 'os'):
                    ax.loglog(k, np.where(s[key] > 0, s[key], np.nan), mk + '-', ms=3, label=labels[key])
                E0 = max(s[keys[0]][0], 1e-12)
                for slope, style in ((-5 / 3, '--'), (-3, ':')):
                    ax.loglog(k, E0 * (k / k[0]) ** slope, 'k' + style, lw=1, label=f'$k^{{{slope:.2g}}}$')
                ax.axvspan(1 / a.min_wavelength_km, k[-1] * 1.05, color='0.8', alpha=0.45, lw=0)
                ax.set_title(f'{name}: ' + ('1-D spectra' if row == 0 else 'Helmholtz decomposition'))
                ax.set_xlabel('wavenumber [cycles/km]')
                ax.grid(True, which='both', alpha=0.25)
                if col == 0:
                    ax.set_ylabel('E(k) [(m/s)$^2$/(cycles/km)]')
                    ax.legend(fontsize=8)
        fig.suptitle(f"{r['name']} — {r['filter']} low-pass; grey: λ < {a.min_wavelength_km:g} km (excluded from fits)")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print(f'wrote {a.plot}')


if __name__ == '__main__':
    main()
