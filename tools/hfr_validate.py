#!/usr/bin/env python3
"""hfr_validate.py — known-answer test of the hfr_spectra.py method on the real radar geometry.

Builds synthetic months with a prescribed KE spectrum (k^-5/3 or k^-3) and rotational share
(1, 0.5, 0), contaminated like the real data (K1 tide, 53 h inertial oscillation, 3 cm/s noise),
masked with the *real* gap pattern of a chosen month, and runs the exact analysis used for the
report. Writes a JSON summary that tools/hfr_report.py quotes as the method's error budget.

Example:
  python tools/hfr_validate.py analysis/npz/hfr_202503.npz --out analysis/report/validation.json
"""
import argparse
import json
import os
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hfr_spectra as S  # noqa: E402


def synthetic_month(real, alpha, share, seed, path):
    """Isotropic field with KE ∝ k^-alpha; `share` of the energy rotational, the rest divergent."""
    mask = np.isfinite(real['u'])
    nt, ny, nx = mask.shape
    dx = json.loads(str(real['meta'])).get('dxKm') or 2.0      # grid spacing of the real product [km]
    N = 256
    k1 = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(k1, k1)
    K = np.hypot(KX, KY)
    K[0, 0] = 1.0
    rng = np.random.default_rng(seed)
    lmax = max(300.0, 2.0 * max(nx, ny) * dx)                         # energy-containing scale ≥ footprint
    amp = np.where(K > 2 * np.pi / lmax, K ** (-(alpha + 3) / 2), 0.0)  # |ψ̂|, |φ̂| → KE ∝ k^-alpha
    ph1, ph2 = rng.uniform(0, 2 * np.pi, (2,) + K.shape)
    om = rng.normal(0, 2 * np.pi / 288, K.shape)                         # ~12-day evolution
    u = np.empty((nt, ny, nx))
    v = np.empty_like(u)
    for t in range(nt):
        ps = amp * np.exp(1j * (ph1 - om * t)) * np.sqrt(share)
        ch = amp * np.exp(1j * (ph2 - om * t)) * np.sqrt(1 - share)
        u[t] = np.real(np.fft.ifft2(-1j * KY * ps + 1j * KX * ch))[:ny, :nx]
        v[t] = np.real(np.fft.ifft2(1j * KX * ps + 1j * KY * ch))[:ny, :nx]
    s = 0.08 / np.sqrt(np.mean(u ** 2 + v ** 2))
    t = np.arange(nt)[:, None, None]
    u = u * s + 0.25 * np.cos(2 * np.pi * t / 23.93) + 0.08 * np.cos(2 * np.pi * t / 53.3) + rng.normal(0, 0.03, u.shape)
    v = v * s + 0.18 * np.cos(2 * np.pi * t / 23.93) + 0.08 * np.sin(2 * np.pi * t / 53.3) + rng.normal(0, 0.03, v.shape)
    u[~mask] = np.nan
    v[~mask] = np.nan
    np.savez(path, u=u.astype(np.float32), v=v.astype(np.float32), lon=real['lon'], lat=real['lat'],
             time=real['time'], meta=str(real['meta']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mask_npz', help='real month whose gap pattern is imposed')
    ap.add_argument('--realisations', type=int, default=3)
    ap.add_argument('--segment', type=int, default=26, help='segment length in nodes, both directions (default 26)')
    ap.add_argument('--direction', default='combined', choices=['combined', 'zonal', 'meridional'])
    ap.add_argument('--out', default=None, help='write a JSON summary here')
    a = ap.parse_args()
    warnings.simplefilter('ignore', RuntimeWarning)
    real = np.load(a.mask_npz)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for alpha in (5 / 3, 3.0):
            for share in (1.0, 0.5, 0.0):
                key = 'Dpsi' if share > 0 else 'Dphi'
                fits, shares = [], []
                for seed in range(a.realisations):
                    p = os.path.join(tmp, f's{seed}.npz')
                    synthetic_month(real, alpha, share, seed, p)
                    z = S.analyze(p, segment=a.segment, segment_meridional=a.segment)['spectra'][a.direction]
                    fits.append(z['fit'][key][0])
                    shares.append(z['rot_frac'])
                rows.append({'true_slope': -alpha, 'true_rot_share': share, 'component': key,
                             'fitted_slopes': fits, 'fitted_rot_shares': shares})
                print(f"true k^{-alpha:.2f}, rotational share {share:.1f}: {key} slopes "
                      + ', '.join(f'{f:+.2f}' for f in fits) + ' | shares ' + ', '.join(f'{x:.2f}' for x in shares))
    summary = {'mask': os.path.basename(a.mask_npz), 'realisations': a.realisations, 'segment': a.segment,
               'direction': a.direction, 'cases': rows}
    for alpha in (-5 / 3, -3.0):
        f = np.concatenate([r['fitted_slopes'] for r in rows if abs(r['true_slope'] - alpha) < 1e-9])
        summary[f'slope_{alpha:.2f}'] = {'bias': float(f.mean() - alpha), 'scatter': float(f.std(ddof=1))}
    sh = np.concatenate([np.array(r['fitted_rot_shares']) - r['true_rot_share'] for r in rows])
    summary['rot_share'] = {'bias': float(sh.mean()), 'rms_error': float(np.sqrt(np.mean(sh ** 2)))}
    print(json.dumps({k: v for k, v in summary.items() if k != 'cases'}, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w') as fh:
            json.dump(summary, fh, indent=1)


if __name__ == '__main__':
    main()
