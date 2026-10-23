#!/usr/bin/env python3
"""hfr_rotation_test.py — test the rotating-turbulence (QNSE) predictions of Galperin et al.
across latitudes with HF radar: low latitude (GISTDA, Gulf of Thailand, ~13°N) versus mid
latitude (NOAA NCEI CODAR, central California, ~37°N).

QNSE one-dimensional spectra (Galperin et al., IOVWST 2022 presentation, p. 4):
    E_L(k) = (18/55) C_K Π^(2/3) k^-5/3 + 0.0926 f² k^-3
    E_T(k) = (24/55) C_K Π^(2/3) k^-5/3 + 0.24   f² k^-3      (k in rad/m, one-sided)
  * Coriolis range: the k^-3 amplitude depends on f only (not on any flux), so the compensated
    spectra C_L k³/f² and C_T k³/f² must equal 0.0926 and 0.24 at every latitude — an absolute,
    parameter-free test.
  * Transition where the two terms are equal: k_t = [0.0926 f² / ((18/55) C_K Π^(2/3))]^(3/4)
    ∝ f^(3/2) Π^(-1/2) (Zeman scaling). As f → 0, k_t → 0 and the Coriolis range vanishes.

ε is measured independently of the spectra from the third-order structure function
  V(r) = ⟨δu_L (δu_L² + δu_T²)⟩ = 2 ε r            (Lindborg 1999; Balwada et al. 2016)
with V > 0 meaning an inverse (upscale) energy flux. Confidence intervals: block bootstrap
over days. All quantities use the rotational (Bühler et al. 2014) part of the subtidal spectrum.

Example:
  python tools/hfr_rotation_test.py
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hfr_spectra as S  # noqa: E402

OMEGA = 7.2921e-5
QNSE_L, QNSE_T = 0.0926, 0.24          # Coriolis-range amplitudes of E_L, E_T (in units of f² k^-3)
C_K = 1.6                              # Kolmogorov constant used for the transition estimate (assumed)


def transition_km(f, flux):
    """QNSE transition wavelength where the k^-5/3 and f² k^-3 terms of E_L are equal [km]."""
    k_t = (QNSE_L * f ** 2 / ((18 / 55) * C_K * flux ** (2 / 3))) ** 0.75
    return 2 * math.pi / k_t / 1e3


def latitude_for_transition(lam_km, flux):
    """Latitude below which the transition wavelength exceeds lam_km (Coriolis range unresolved)."""
    k_t = 2 * math.pi / (lam_km * 1e3)
    f = math.sqrt(k_t ** (4 / 3) * (18 / 55) * C_K * flux ** (2 / 3) / QNSE_L)
    return math.degrees(math.asin(min(1.0, f / (2 * OMEGA))))


def subtidal_fields(path, kind='harmonic', cutoff_h=72.0, half_h=120.0):
    """kind='harmonic': remove tidal constituents only (all non-tidal motion kept);
    kind='lanczos': subtidal (> 72 h) only."""
    d = np.load(path)
    meta = json.loads(str(d['meta']))
    if kind == 'harmonic':
        u, v = S.detide_harmonic(d['u'].astype(float), meta['stepHours']), S.detide_harmonic(d['v'].astype(float), meta['stepHours'])
    else:
        w = S.lanczos_weights(cutoff_h, half_h, meta['stepHours'])
        u, v = S.lowpass(d['u'].astype(float), w), S.lowpass(d['v'].astype(float), w)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        u, v = u - np.nanmean(u, axis=0), v - np.nanmean(v, axis=0)
    return u, v, meta, float(np.mean(d['lat']))


def third_order(u, v, dx, max_lag):
    """Per-lag sums of δu_L(δu_L² + δu_T²) and pair counts, per hour (for block bootstrap)."""
    nt = u.shape[0]
    S3 = np.zeros((nt, max_lag)); N = np.zeros((nt, max_lag))
    for axis, along, across in ((2, u, v), (1, v, u)):
        for lag in range(1, max_lag + 1):
            if lag >= along.shape[axis]:
                break
            a0, a1 = [slice(None)] * 3, [slice(None)] * 3
            a0[axis], a1[axis] = slice(lag, None), slice(None, -lag)
            dL = along[tuple(a0)] - along[tuple(a1)]
            dT = across[tuple(a0)] - across[tuple(a1)]
            ok = np.isfinite(dL) & np.isfinite(dT)
            val = np.where(ok, dL * (dL ** 2 + dT ** 2), 0.0)
            S3[:, lag - 1] += val.reshape(nt, -1).sum(axis=1)
            N[:, lag - 1] += ok.reshape(nt, -1).sum(axis=1)
    return np.arange(1, max_lag + 1) * dx, S3, N


def eps_from_V(r, S3, N, rmin, rmax, n_boot=1000, seed=0):
    """ε = V/(2r) averaged over [rmin, rmax]; 95 % CI by bootstrap over daily blocks."""
    sel = (r >= rmin) & (r <= rmax)
    nt = S3.shape[0]
    days = np.arange(nt) // 24
    blocks = [np.where(days == d)[0] for d in np.unique(days)]
    blocks = [b for b in blocks if N[b][:, sel].sum() > 0]

    def est(idx):
        s, n = S3[idx].sum(axis=0), N[idx].sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            V = s / n
        return np.nanmean(V[sel] / (2 * r[sel] * 1e3)), V      # r in m → ε in m²/s³

    eps, V = est(np.concatenate(blocks))
    rng = np.random.default_rng(seed)
    boots = [est(np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]))[0] for _ in range(n_boot)]
    return eps, np.nanpercentile(boots, [2.5, 97.5]), V


def compensated(k_cpkm, E_cpkm, f):
    """E k³ / f² (dimensionless) with E in (m/s)² per cycle/km and k in cycles/km."""
    k_si = 2 * np.pi * k_cpkm / 1e3                    # rad/m
    E_si = E_cpkm * 1e3 / (2 * np.pi)                  # m³/s²
    return k_si, E_si * k_si ** 3 / f ** 2


def null_sigma(path, segment, floor, rmax_km, rms_real, kind, n=3):
    """Spread of the ε estimator on synthetic no-flux months with this product's gap mask,
    rescaled to the real subtidal rms (third moments scale with u³)."""
    import tempfile
    import hfr_validate as Vd
    real = np.load(path)
    vals = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(n):
            q = os.path.join(tmp, 'null.npz')
            Vd.synthetic_month(real, 5 / 3, 1.0, 100 + seed, q)
            u, v, meta, _ = subtidal_fields(q, kind)
            rms = math.sqrt(np.nanmean(u ** 2 + v ** 2))
            r, S3, N = third_order(u, v, meta['dxKm'], int(rmax_km / meta['dxKm']) + 1)
            e, _, _ = eps_from_V(r, S3, N, floor, rmax_km, n_boot=1)
            vals.append(e * (rms_real / rms) ** 3)
    return float(np.std(vals, ddof=1))


def noise_floor(k, E):
    """White-noise level from the highest-wavenumber bins (λ ≲ 3 grid lengths), conservative minimum."""
    hi = (1 / k) <= 3.2 * (1 / k[-1])
    return float(E[hi].min())


def dataset(label, paths, segment, floor, rmax_km, kind='harmonic'):
    """Average spectra over months/files (white-noise floor removed); ε from pooled third-order
    structure functions, with a null-calibrated noise level."""
    specs, S3s, Ns, lats, r = [], [], [], [], None
    for p in paths:
        res = S.analyze(p, lowpass_kind=kind, segment=segment, segment_meridional=segment, min_wl_km=floor)
        c = res['spectra'].get('combined')
        if not c:
            continue
        specs.append(c)
        u, v, meta, lat = subtidal_fields(p, kind)
        lats.append(lat)
        lag_max = int(rmax_km / meta['dxKm']) + 1
        r, S3, N = third_order(u, v, meta['dxKm'], lag_max)
        S3s.append(S3); Ns.append(N)
    k = specs[0]['k']
    Dpsi = np.mean([s['Dpsi'] for s in specs], axis=0)
    Dtot = np.mean([s['Dpsi'] + s['Dphi'] for s in specs], axis=0)
    lat = float(np.mean(lats))
    f = 2 * OMEGA * math.sin(math.radians(lat))
    eps, eps_ci, V = eps_from_V(r, np.concatenate(S3s), np.concatenate(Ns), floor, rmax_km)
    u0, v0, _, _ = subtidal_fields(paths[0], kind)
    rms_real = math.sqrt(np.nanmean(u0 ** 2 + v0 ** 2))
    sig_null = null_sigma(paths[0], segment, floor, rmax_km, rms_real, kind) / math.sqrt(len(specs))
    CL = np.mean([s['CL'] for s in specs], axis=0)
    CT = np.mean([s['CT'] for s in specs], axis=0)
    CL, CT = CL - noise_floor(k, CL), CT - noise_floor(k, CT)          # radar white noise
    k_om = math.sqrt(f ** 3 / abs(eps)) if eps else math.nan          # rad/m
    lam_om = 2 * math.pi / k_om / 1e3 if np.isfinite(k_om) else math.nan  # km
    band = (1 / k) >= floor
    k_si, comp = compensated(k, np.where(Dpsi > 0, Dpsi, np.nan), f)
    slope = S.fit_power_law(k, Dpsi, floor)
    # Flux implied by the spectral level if these scales were QNSE's k^-5/3 range (C_K = 1.6)
    kb, Eb = 2 * np.pi * k[band] / 1e3, CL[band] * 1e3 / (2 * np.pi)
    pi_spec = float(np.nanmedian((np.where(Eb > 0, Eb, np.nan) / ((18 / 55) * C_K * kb ** (-5 / 3))) ** 1.5)) if band.any() else math.nan
    _, compL = compensated(k, CL, f)
    _, compT = compensated(k, CT, f)
    return {'label': label, 'kind': kind, 'lat': lat, 'f': f, 'months': len(specs), 'k': k, 'k_si': k_si, 'Dpsi': Dpsi,
            'CL': CL, 'CT': CT, 'compL': compL, 'compT': compT, 'sig_null': sig_null, 'pi_spec': pi_spec,
            'AL': float(np.nanmedian(compL[band])), 'AT': float(np.nanmedian(compT[band])),
            'slopeL': S.fit_power_law(k, CL, floor), 'slopeT': S.fit_power_law(k, CT, floor),
            'Dtot': Dtot, 'comp': comp, 'band': band, 'eps': eps, 'eps_ci': eps_ci, 'r': r, 'V': V,
            'k_omega': k_om, 'lambda_omega_km': lam_om, 'slope': slope, 'floor': floor,
            'C_f': float(np.nanmedian(comp[band])) if band.any() else math.nan}


def main():
    warnings.simplefilter('ignore', RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz-dir', default=os.path.join(ROOT, 'analysis', 'npz'))
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'analysis', 'rotation'))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    summ = os.path.join(ROOT, 'analysis', 'report', 'summary.json')
    used = {m['ym'] for m in json.load(open(summ))['months'] if m['used']} if os.path.exists(summ) else None
    gistda = sorted(p for p in glob.glob(os.path.join(a.npz_dir, 'hfr_*.npz'))
                    if re.fullmatch(r'hfr_(\d{6})\.npz', os.path.basename(p))
                    and (used is None or re.search(r'(\d{6})', os.path.basename(p)).group(1) in used))
    specs = []
    if gistda:
        specs.append(('GISTDA, Gulf of Thailand (13°N), 15 months', gistda, 26, 20.0, 40.0))
    for name, label, seg, floor, rmax in (('hfr_ncei_ccal_202012', 'NOAA central California (36°N), Dec 2020', 32, 19.8, 120.0),
                                          ('hfr_ncei_mbay_202012', 'NOAA Monterey Bay (36°N), Dec 2020', 16, 6.6, 20.0)):
        p = os.path.join(a.npz_dir, name + '.npz')
        if os.path.exists(p):
            specs.append((label, [p], seg, floor, rmax))
    # Primary: tides removed harmonically (all non-tidal motion). Secondary: subtidal (> 72 h) only.
    sets = [dataset(*sp, kind='harmonic') for sp in specs]
    sub = [dataset(*sp, kind='lanczos') for sp in specs]
    if not sets:
        sys.exit('no datasets')

    # ── console summary ──
    print(f"QNSE Coriolis-range amplitudes: C_L k³/f² = {QNSE_L}, C_T k³/f² = {QNSE_T} (latitude-independent)")
    print(f"{'dataset':44s} {'lat':>5s} {'f·1e5':>6s} | {'C_L slope':>9s} {'C_T slope':>9s} | {'A_L=C_Lk³/f²':>12s} {'A_T':>7s} "
          f"{'A_T/A_L':>7s} | {'ε·1e9 (null σ)':>15s}")
    for s_ in sets + sub:
        print(f"{s_['label'][:34] + ' [' + s_['kind'][:4] + ']':44s} {s_['lat']:5.1f} {s_['f'] * 1e5:6.2f} | {s_['slopeL'][0]:+9.2f} {s_['slopeT'][0]:+9.2f} | "
              f"{s_['AL']:12.4f} {s_['AT']:7.4f} {s_['AT'] / s_['AL']:7.2f} | {s_['eps'] * 1e9:+6.2f} ({s_['sig_null'] * 1e9:.2f})")
    print('\nFlux consistency (tides removed): Π from V(r) vs Π implied by the k^-5/3 level of E_L (QNSE, C_K = 1.6)')
    for s_ in sets:
        print(f"  {s_['label'][:44]:44s} Π_V = {s_['eps'] * 1e9:+.2f}e-9 (null σ {s_['sig_null'] * 1e9:.2f})   Π_spec = {s_['pi_spec'] * 1e9:.1f}e-9   ratio {s_['pi_spec'] / s_['eps']:.0f}×")
    print(f"{'QNSE prediction':44s} {'':5s} {'':6s} | {-3:+9.2f} {-3:+9.2f} | {QNSE_L:12.4f} {QNSE_T:7.4f} {QNSE_T / QNSE_L:7.2f} |")
    print('\nQNSE transition wavelength λ_t [km] (C_K = 1.6) and the latitude below which λ_t > 20 km:')
    for flux in (1e-9, 1e-8, 1e-7):
        lams = ', '.join(f'{la}°N: {transition_km(2 * OMEGA * math.sin(math.radians(la)), flux):.1f}' for la in (2, 5, 13, 20, 37))
        print(f'  Π = {flux:.0e} m²/s³ → {lams}   | Coriolis range unresolved by radar (λ_t > 20 km) for |lat| < '
              f'{latitude_for_transition(20, flux):.1f}°')

    # ── figures ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for s_, t_, c in zip(sets, sub, colors):
        band = s_['band']
        lam = 1 / s_['k']
        for key, mk, ls in (('compL', 'o', '-'), ('compT', 's', '--')):
            ax[0].semilogx(lam[band], s_[key][band], mk + ls, ms=4.5, color=c,
                           label=f"{s_['label'].split(',')[0]} {'$C_L$' if key == 'compL' else '$C_T$'}")
            ax[0].semilogx(lam[band], t_[key][band], mk + ':', ms=2.5, color=c, alpha=0.45)
    ax[0].axhline(QNSE_L, color='k', ls='-', lw=1); ax[0].axhline(QNSE_T, color='k', ls='--', lw=1)
    ax[0].text(ax[0].get_xlim()[0] * 1.1 if ax[0].get_xlim()[0] > 0 else 3, QNSE_T * 1.08, 'QNSE $C_T$: 0.24', fontsize=8)
    ax[0].text(ax[0].get_xlim()[0] * 1.1 if ax[0].get_xlim()[0] > 0 else 3, QNSE_L * 1.08, 'QNSE $C_L$: 0.0926', fontsize=8)
    ax[0].set_yscale('log'); ax[0].invert_xaxis()
    ax[0].set_xlabel('wavelength [km]'); ax[0].set_ylabel(r'$E\,k^3/f^2$')
    ax[0].set_title('QNSE Coriolis-range test: $Ek^3/f^2$ must be flat at 0.0926 / 0.24\n'
                    '(bold: tides removed; dotted: subtidal > 72 h)', fontsize=9.5)
    lats = np.linspace(0.5, 40, 200)
    for flux, st in ((1e-9, ':'), (1e-8, '-'), (1e-7, '--')):
        ax[1].semilogy(lats, [transition_km(2 * OMEGA * math.sin(math.radians(x)), flux) for x in lats], 'k' + st,
                       label=f'Π = {flux:.0e} m²/s³')
    ax[1].axhspan(20, 200, color='#1f77b4', alpha=0.08, label='radar-resolved band (≥ 20 km)')
    for s_, c in zip(sets, colors):
        ax[1].axvline(s_['lat'], color=c, lw=1.5, alpha=0.7)
    ax[1].set_xlabel('latitude [°]'); ax[1].set_ylabel(r'QNSE transition wavelength $\lambda_t$ [km]')
    ax[1].set_title('Where the Coriolis range must vanish (f → 0)')
    ax[1].set_ylim(0.01, 1e4)
    for s_, c in zip(sets, colors):
        ax[2].semilogx(s_['r'], s_['V'] / (2 * s_['r'] * 1e3) * 1e9, '-', color=c, label=s_['label'].split(',')[0])
        ax[2].axhspan(-2 * s_['sig_null'] * 1e9, 2 * s_['sig_null'] * 1e9, color=c, alpha=0.08)
    ax[2].axhline(0, color='k', lw=0.8)
    ax[2].set_xlabel('separation r [km]'); ax[2].set_ylabel(r'$V(r)/2r$  [$10^{-9}$ m$^2$/s$^3$]')
    ax[2].set_title(r'Energy flux $V=\langle\delta u_L(\delta u_L^2+\delta u_T^2)\rangle$; band: ±2σ null noise')
    for x_ in ax:
        x_.grid(True, which='both', alpha=0.25)
        x_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'rotation_test.png'), dpi=140)

    fin = lambda x: None if x is None or not np.isfinite(x) else float(x)
    out = {'qnse': {'A_L': QNSE_L, 'A_T': QNSE_T, 'C_K': C_K,
                    'source': 'Galperin et al., IOVWST 2022 presentation, p. 4'},
           'transition': {f'{flux:.0e}': {'lat_20km': latitude_for_transition(20, flux),
                                          'lambda_km': {str(la): transition_km(2 * OMEGA * math.sin(math.radians(la)), flux)
                                                        for la in (2, 5, 13, 20, 37)}} for flux in (1e-9, 1e-8, 1e-7)},
           'datasets': [{'label': s_['label'], 'lat': s_['lat'], 'f': s_['f'], 'months': s_['months'],
                         'A_L': s_['AL'], 'A_T': s_['AT'], 'slope_L': fin(s_['slopeL'][0]), 'slope_T': fin(s_['slopeT'][0]),
                         'slope_psi': fin(s_['slope'][0]), 'eps': s_['eps'], 'eps_null_sigma': s_['sig_null'],
                         'pi_spec': fin(s_['pi_spec']),
                         'floor_km': s_['floor'], 'fit_wavelengths_km': [round(1 / x, 1) for x in s_['k'][s_['band']]]}
                        for s_ in sets],
           'subtidal': [{'label': s_['label'], 'A_L': s_['AL'], 'A_T': s_['AT'], 'slope_L': fin(s_['slopeL'][0]),
                         'slope_T': fin(s_['slopeT'][0]), 'eps': s_['eps'], 'eps_null_sigma': s_['sig_null']} for s_ in sub],
           'figure': 'analysis/rotation/rotation_test.png'}
    with open(os.path.join(a.out_dir, 'rotation_test.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {a.out_dir}/rotation_test.png and rotation_test.json")


if __name__ == '__main__':
    main()
