#!/usr/bin/env python3
"""hfr_latitude_test.py — QNSE rotating-turbulence test across many HF radar areas and latitudes.

Inputs: every analysis/npz/rtv_*.npz (official HFRNet total vectors, cut by tools/hfr_rtv.py) and
the GISTDA Gulf of Thailand months (pooled, 13°N). All are total-vector products (like .tuv).

Per area (tides removed harmonically = primary; subtidal > 72 h = secondary; white-noise floor
removed; only wavelengths ≥ --floor-km, above the products' ~10 km search radius):
  * slopes of C_L, C_T and the rotational D_ψ
  * QNSE Coriolis-range amplitudes A_L = C_L k³/f², A_T = C_T k³/f²  (QNSE: 0.0926, 0.24 everywhere)
  * spectral level E_ref = C_L + C_T at a fixed wavelength (--ref-km)
Across areas:
  * exponent p in E_ref ∝ f^p (bootstrap 95 % CI over areas): QNSE Coriolis range ⇒ p = 2;
    an f-independent (Kolmogorov / flux-controlled) range ⇒ p ≈ 0.

Example:
  python tools/hfr_latitude_test.py
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hfr_spectra as S  # noqa: E402

OMEGA = 7.2921e-5
QNSE_L, QNSE_T = 0.0926, 0.24


def noise_floor(k, E):
    hi = (1 / k) <= 3.2 * (1 / k[-1])
    return float(np.nanmin(E[hi]))


def spectra(paths, segment, floor, kind):
    CL, CT, Dp, lat, n, sub = [], [], [], [], 0, []
    for p in paths:
        r = S.analyze(p, lowpass_kind=kind, segment=segment, segment_meridional=segment, min_wl_km=floor)
        c = r['spectra'].get('combined')
        if not c or c['nseg'] < 50:
            continue
        CL.append(c['CL']); CT.append(c['CT']); Dp.append(c['Dpsi'])
        lat.append(float(np.mean(np.load(p)['lat']))); n += c['nseg']; sub.append(r['subtidal_frac'])
        k = c['k']
    if not CL:
        return None
    CL, CT, Dp = np.mean(CL, 0), np.mean(CT, 0), np.mean(Dp, 0)
    CL, CT = CL - noise_floor(k, CL), CT - noise_floor(k, CT)
    return dict(k=k, CL=CL, CT=CT, Dpsi=Dp, lat=float(np.mean(lat)), nseg=n, frac=float(np.mean(sub)))


def area(args):
    label, paths, segment, floor, ref_km = args
    warnings.simplefilter('ignore', RuntimeWarning)
    out = {'label': label}
    for kind in ('harmonic', 'lanczos'):
        s = spectra(paths, segment, floor, kind)
        if s is None:
            return None
        k, band = s['k'], (1 / s['k']) >= floor
        f = 2 * OMEGA * math.sin(math.radians(abs(s['lat'])))
        ks = 2 * np.pi * k / 1e3
        comp = lambda E: E * 1e3 / (2 * np.pi) * ks ** 3 / f ** 2
        pos = lambda E: np.where(E > 0, E, np.nan)
        Etot = pos(s['CL'] + s['CT'])
        E_ref = float(np.exp(np.interp(math.log(1 / ref_km), np.log(k), np.log(Etot), left=np.nan, right=np.nan)))
        out[kind] = {
            'lat': s['lat'], 'f': f, 'segments': s['nseg'], 'bins': int(band.sum()),
            'slope_L': S.fit_power_law(k, pos(s['CL']), floor)[0], 'slope_T': S.fit_power_law(k, pos(s['CT']), floor)[0],
            'slope_psi': S.fit_power_law(k, pos(s['Dpsi']), floor)[0],
            'A_L': float(np.nanmedian(comp(pos(s['CL']))[band])), 'A_T': float(np.nanmedian(comp(pos(s['CT']))[band])),
            'E_ref': E_ref, 'kept_frac': s['frac'],
            'k': k.tolist(), 'compL': comp(pos(s['CL'])).tolist(), 'compT': comp(pos(s['CT'])).tolist(), 'band': band.tolist(),
        }
    out['lat'] = out['harmonic']['lat']
    return out


def exponent(fs, Es, n_boot=4000, seed=0):
    x, y = np.log(fs), np.log(Es)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    p = np.polyfit(x, y, 1)[0]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0:
            bs.append(np.polyfit(x[i], y[i], 1)[0])
    return p, np.percentile(bs, [2.5, 97.5]), int(ok.sum())


def main():
    warnings.simplefilter('ignore', RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz-dir', default=os.path.join(ROOT, 'analysis', 'npz'))
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'analysis', 'latitude'))
    ap.add_argument('--floor-km', type=float, default=22.0, help='minimum fitted wavelength (2.2 × 10 km search radius)')
    ap.add_argument('--ref-km', type=float, default=50.0, help='wavelength for the cross-latitude energy level')
    ap.add_argument('--segment', type=int, default=24, help='segment length in cells for the 6 km RTV areas')
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--min-months', type=int, default=3, help='sites need at least this many months')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # Group monthly RTV areas into sites: same region, centres within 1° lat / 1.5° lon.
    pat = re.compile(r'rtv_([a-z]+)_(\d{6})_([\d.]+)([NS])_([\d.]+)([EW])\.npz')
    sites = []
    for p in sorted(glob.glob(os.path.join(a.npz_dir, 'rtv_*.npz'))):
        m = pat.fullmatch(os.path.basename(p))
        if not m:
            continue
        reg, ym = m.group(1), m.group(2)
        la = float(m.group(3)) * (1 if m.group(4) == 'N' else -1)
        lo = float(m.group(5)) * (1 if m.group(6) == 'E' else -1)
        g = next((g for g in sites if g['region'] == reg and abs(g['lat'] - la) <= 1.0 and abs(g['lon'] - lo) <= 1.5), None)
        if g is None:
            g = {'region': reg, 'lat': la, 'lon': lo, 'paths': [], 'months': set()}
            sites.append(g)
        if ym not in g['months']:
            g['paths'].append(p); g['months'].add(ym)
    jobs = []
    for g in sites:
        if len(g['months']) < a.min_months:
            continue
        lab = f"{g['region'].upper()} {abs(g['lat']):.1f}°{'N' if g['lat'] >= 0 else 'S'} {abs(g['lon']):.1f}°{'W' if g['lon'] < 0 else 'E'} ({len(g['months'])} mo)"
        jobs.append((lab, g['paths'], a.segment, a.floor_km, a.ref_km))
    print(f"{len(jobs)} HFRNet sites with ≥ {a.min_months} months (of {len(sites)} candidate sites)")
    summ = os.path.join(ROOT, 'analysis', 'report', 'summary.json')
    if os.path.exists(summ):
        used = {m['ym'] for m in json.load(open(summ))['months'] if m['used']}
        g = sorted(p for p in glob.glob(os.path.join(a.npz_dir, 'hfr_*.npz'))
                   if re.fullmatch(r'hfr_(\d{6})\.npz', os.path.basename(p)) and os.path.basename(p)[4:10] in used)
        if g:
            jobs.append(('GISTDA Gulf of Thailand (15 months)', g, 26, 20.0, a.ref_km))
    with ProcessPoolExecutor(a.jobs) as ex:
        areas = [r for r in ex.map(area, jobs) if r]
    areas.sort(key=lambda r: r['lat'])

    print(f"{'area':38s} {'lat':>5s} {'bins':>4s} | {'slope L':>7s} {'T':>6s} {'ψ':>6s} | {'A_L/QNSE':>8s} {'A_T/QNSE':>8s} | "
          f"{'E(50km)':>8s} | subtidal slope ψ")
    for r in areas:
        h, l = r['harmonic'], r['lanczos']
        print(f"{r['label'][:38]:38s} {r['lat']:5.1f} {h['bins']:4d} | {h['slope_L']:+7.2f} {h['slope_T']:+6.2f} {h['slope_psi']:+6.2f} | "
              f"{h['A_L'] / QNSE_L:8.3f} {h['A_T'] / QNSE_T:8.3f} | {h['E_ref']:8.3g} | {l['slope_psi']:+.2f}")
    results = {}
    for kind in ('harmonic', 'lanczos'):
        fs = np.array([r[kind]['f'] for r in areas]); Es = np.array([r[kind]['E_ref'] for r in areas])
        p, ci, n = exponent(fs, Es)
        sl = np.array([r[kind]['slope_psi'] for r in areas])
        results[kind] = {'exponent': p, 'ci': ci.tolist(), 'n': n, 'slope_psi_median': float(np.nanmedian(sl)),
                         'slope_psi_iqr': np.nanpercentile(sl, [25, 75]).tolist()}
        print(f"[{kind:8s}] E({a.ref_km:g} km) ∝ f^p: p = {p:+.2f} (95 % CI {ci[0]:+.2f} … {ci[1]:+.2f}, n = {n} areas) "
              f"| QNSE Coriolis range: p = 2 | median D_ψ slope {np.nanmedian(sl):+.2f}")

    # Robustness of the exponent: drop the Gulf Stream area (inhomogeneous jet) and/or GISTDA (other processing)
    robust = []
    gulf_stream = lambda r: 'USEGC' in r['label'].upper() and 35 < r['lat'] < 36.5
    for name, keep in (('all areas', lambda r: True), ('without Gulf Stream', lambda r: not gulf_stream(r)),
                       ('HFRNet only', lambda r: not r['label'].startswith('GISTDA')),
                       ('HFRNet only, no Gulf Stream', lambda r: not r['label'].startswith('GISTDA') and not gulf_stream(r))):
        rs = [r for r in areas if keep(r)]
        for kind in ('harmonic', 'lanczos'):
            p_, ci_, n_ = exponent(np.array([r[kind]['f'] for r in rs]), np.array([r[kind]['E_ref'] for r in rs]))
            robust.append({'case': name, 'kind': kind, 'p': float(p_), 'ci': ci_.tolist(), 'n': n_})
    ratios = {kind: {'A_L': float(np.nanmedian([r[kind]['A_L'] / QNSE_L for r in areas])),
                     'A_T': float(np.nanmedian([r[kind]['A_T'] / QNSE_T for r in areas]))} for kind in ('harmonic', 'lanczos')}

    # ── figure ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    lats = np.array([r['lat'] for r in areas])
    for kind, mk, al in (('harmonic', 'o', 1.0), ('lanczos', 's', 0.45)):
        AL = np.array([r[kind]['A_L'] for r in areas]); AT = np.array([r[kind]['A_T'] for r in areas])
        ax[0].semilogy(lats, AL, mk, color='#1f77b4', alpha=al, label=f'$C_L k^3/f^2$ ({kind})')
        ax[0].semilogy(lats, AT, mk, color='#d62728', alpha=al, label=f'$C_T k^3/f^2$ ({kind})')
        ax[1].plot(lats, [r[kind]['slope_psi'] for r in areas], mk, color='k', alpha=al, label=f'$D_\\psi$ slope ({kind})')
        fs = np.array([r[kind]['f'] for r in areas]); Es = np.array([r[kind]['E_ref'] for r in areas])
        ax[2].loglog(fs, Es, mk, color='#2ca02c', alpha=al,
                     label=f"{kind}: p = {results[kind]['exponent']:+.2f} [{results[kind]['ci'][0]:+.2f}, {results[kind]['ci'][1]:+.2f}]")
    ax[0].axhline(QNSE_L, color='#1f77b4', ls='--'); ax[0].axhline(QNSE_T, color='#d62728', ls='--')
    ax[0].set_title('QNSE Coriolis-range amplitude (dashed) vs observed'); ax[0].set_xlabel('latitude [°]')
    for y, st in ((-5 / 3, '--'), (-3, ':')):
        ax[1].axhline(y, color='0.4', ls=st, lw=1)
    ax[1].set_title(f'Rotational slope, λ ≥ {a.floor_km:g} km (−5/3 dashed, −3 dotted)'); ax[1].set_xlabel('latitude [°]')
    fs = np.array([r['harmonic']['f'] for r in areas]); Es = np.array([r['harmonic']['E_ref'] for r in areas])
    ff = np.linspace(fs.min() * 0.9, fs.max() * 1.1, 10)
    ref = np.exp(np.nanmean(np.log(Es) - 2 * np.log(fs)))
    ax[2].loglog(ff, ref * ff ** 2, 'k--', lw=1, label='QNSE: $E \\propto f^2$')
    for r in areas:
        ax[2].annotate(f"{r['lat']:.0f}°", (r['harmonic']['f'], r['harmonic']['E_ref']), fontsize=7, alpha=0.7)
    ax[2].set_xlabel('f [s$^{-1}$]'); ax[2].set_ylabel(f'$C_L+C_T$ at λ = {a.ref_km:g} km')
    ax[2].set_title('Does energy scale with f²?')
    for x_ in ax:
        x_.grid(True, which='both', alpha=0.25)
        x_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'latitude_test.png'), dpi=140)

    fin = lambda x: None if x is None or not np.isfinite(x) else float(x)
    out = {'areas': [{'label': r['label'], 'lat': r['lat'],
                      **{f'{kind}_{k_}': fin(r[kind][k_]) for kind in ('harmonic', 'lanczos')
                         for k_ in ('slope_L', 'slope_T', 'slope_psi', 'A_L', 'A_T', 'E_ref')},
                      'bins': r['harmonic']['bins'], 'segments': r['harmonic']['segments']} for r in areas],
           'fit': results, 'robust': robust, 'median_ratio_to_qnse': ratios, 'floor_km': a.floor_km, 'ref_km': a.ref_km, 'qnse': {'A_L': QNSE_L, 'A_T': QNSE_T},
           'figure': 'analysis/latitude/latitude_test.png'}
    with open(os.path.join(a.out_dir, 'latitude_test.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'wrote {a.out_dir}/latitude_test.png and latitude_test.json')


if __name__ == '__main__':
    main()
