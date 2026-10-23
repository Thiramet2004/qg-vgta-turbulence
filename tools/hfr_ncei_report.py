#!/usr/bin/env python3
"""hfr_ncei_report.py — Galperin-style spectral analysis of NCEI HF radar total-vector months
(produced by tools/hfr_ncei.py), plus the data file behind analysis.html.

For each network:
  * subtidal fields (Lanczos 72 h; removes tides and the ~20 h inertial band at 36–38°N)
  * 1-D longitudinal/transverse spectra C_L, C_T and the Bühler et al. (2014) rotational/divergent
    split D_ψ, D_φ (zonal and meridional transects averaged)
  * power-law fits only above the product's own smoothing scale: λ ≥ 2.2 × search radius
  * known-answer validation on that network's real gap mask (tools/hfr_validate.py)
  * figures: footprint (mean current, data availability) and spectra with fitted slopes

Writes analysis/ncei/<name>_*.png and analysis/web/analysis_data.js (also folds in the GISTDA
seasonal summary from analysis/report/ when present).

Example:
  python tools/hfr_ncei_report.py
"""
import argparse
import json
import math
import os
import sys
import tempfile
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hfr_spectra as S  # noqa: E402
import hfr_validate as V  # noqa: E402

NETWORKS = [   # name, label, search radius [km], segment [nodes]
    ('hfr_ncei_ccal_202012', 'Central California (5 + 13 MHz), 6 km', 9.0, 32),
    ('hfr_ncei_mbay_202012', 'Monterey Bay (13 + 25 MHz), 2 km', 3.0, 16),
    ('hfr_ncei_sfb_202012', 'San Francisco Bay (42 MHz), 1 km', 1.5, 10),
]


def validate(path, segment, floor, realisations=3):
    """Single-month bias/scatter of the D_ψ (or D_φ) slope and rotational-share error on this mask."""
    real = np.load(path)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for alpha in (5 / 3, 3.0):
            fits, shares = [], []
            for share in (1.0, 0.5):
                for seed in range(realisations):
                    p = os.path.join(tmp, 's.npz')
                    V.synthetic_month(real, alpha, share, seed, p)
                    c = S.analyze(p, segment=segment, segment_meridional=segment, min_wl_km=floor)['spectra']['combined']
                    if c:
                        fits.append(c['fit']['Dpsi'][0])
                        shares.append(c['rot_frac'] - share)
            f = np.array(fits)
            out[f'{-alpha:.2f}'] = {'bias': float(np.nanmean(f) + alpha), 'scatter': float(np.nanstd(f, ddof=1))}
            out.setdefault('share_err', []).extend(shares)
    out['share_rms'] = float(np.sqrt(np.nanmean(np.square(out.pop('share_err')))))
    return out


def figures(name, label, path, r, floor, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = np.load(path)
    u, v, lon, lat = d['u'], d['v'], d['lon'], d['lat']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        avail = np.isfinite(u).mean(axis=0)
        um, vm = np.nanmean(u, axis=0), np.nanmean(v, axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    X, Y = np.meshgrid(lon, lat)
    asp = 1 / math.cos(math.radians(float(lat.mean())))
    im = ax[0].pcolormesh(lon, lat, np.where(avail > 0, avail, np.nan), cmap='viridis', vmin=0, vmax=1, shading='auto')
    fig.colorbar(im, ax=ax[0], label='fraction of hours with a vector')
    ax[0].set_title('Data availability')
    step = max(1, int(round(len(lon) / 35)))
    good = avail > 0.5
    ax[1].quiver(X[::step, ::step], Y[::step, ::step], np.where(good, um, np.nan)[::step, ::step],
                 np.where(good, vm, np.nan)[::step, ::step], np.hypot(um, vm)[::step, ::step], cmap='magma_r', width=0.003)
    ax[1].set_title('Monthly mean current (nodes with > 50 % availability)')
    for a in ax:
        a.set_aspect(asp)
        a.set_xlabel('longitude'); a.set_ylabel('latitude')
    fig.suptitle(f'{label} — December 2020', fontsize=11)
    fig.tight_layout()
    f1 = os.path.join(out_dir, f'{name}_footprint.png')
    fig.savefig(f1, dpi=130)
    plt.close(fig)
    return os.path.basename(f1)


def spectra_figure(name, label, res, floor, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = res['spectra']['combined']
    k = c['k']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    lab = {'CL': '$C_L$', 'CT': '$C_T$', 'Dpsi': r'$D_\psi$ rotational', 'Dphi': r'$D_\phi$ divergent'}
    band = (1 / k) >= floor
    for ax, keys in zip(axes, (('CL', 'CT'), ('Dpsi', 'Dphi'))):
        for key, mk in zip(keys, 'os'):
            E = c[key]
            ax.loglog(k, np.where(E > 0, E, np.nan), mk + '-', ms=3.5, label=lab[key])
            neg = E <= 0
            if neg.any():
                ax.loglog(k[neg], -E[neg], mk, mfc='none', ms=5, color='0.4')
            s, se, n = c['fit'][key]
            if np.isfinite(s) and band.sum() >= 2:
                kk = k[band]
                A = np.exp(np.mean(np.log(np.where(E[band] > 0, E[band], np.nan)) - s * np.log(kk)))
                ax.loglog(kk, A * kk ** s, lw=2.2, alpha=0.6, label=f'fit {s:+.2f}' + (f' ± {se:.2f}' if np.isfinite(se) else ''))
        E0 = max(c[keys[0]][0], 1e-12)
        for slope, style in ((-5 / 3, '--'), (-3, ':')):
            ax.loglog(k, E0 * (k / k[0]) ** slope, 'k' + style, lw=1, label=f'$k^{{{slope:.2g}}}$')
        ax.axvspan(1 / floor, k[-1] * 1.05, color='0.85', alpha=0.5, lw=0)
        ax.set_xlabel('wavenumber [cycles/km]')
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=7.5)
        sec = ax.secondary_xaxis('top', functions=(lambda x: 1 / np.maximum(x, 1e-9), lambda x: 1 / np.maximum(x, 1e-9)))
        sec.set_xlabel('wavelength [km]')
    axes[0].set_ylabel('E(k) [(m/s)$^2$/(cycles/km)]')
    axes[0].set_title('1-D spectra (subtidal)')
    axes[1].set_title('Helmholtz decomposition')
    fig.suptitle(f'{label}: grey = λ < {floor:.1f} km (below the product smoothing scale; not fitted)', fontsize=10)
    fig.tight_layout()
    f = os.path.join(out_dir, f'{name}_spectra.png')
    fig.savefig(f, dpi=130)
    plt.close(fig)
    return os.path.basename(f)


def main():
    warnings.simplefilter('ignore', RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz-dir', default=os.path.join(ROOT, 'analysis', 'npz'))
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'analysis', 'ncei'))
    ap.add_argument('--web', default=os.path.join(ROOT, 'analysis', 'web', 'analysis_data.js'))
    ap.add_argument('--realisations', type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.web), exist_ok=True)

    networks = []
    for name, label, r, seg in NETWORKS:
        path = os.path.join(a.npz_dir, name + '.npz')
        if not os.path.exists(path):
            print(f'{name}: missing, skipped')
            continue
        floor = 2.2 * r
        res = S.analyze(path, segment=seg, segment_meridional=seg, min_wl_km=floor)
        c = res['spectra'].get('combined')
        meta = res['meta']
        if not c:
            print(f'{name}: too few segments')
            continue
        val = validate(path, seg, floor, a.realisations)
        foot = figures(name, label, path, r, floor, a.out_dir)
        spec = spectra_figure(name, label, res, floor, a.out_dir)
        fit = {key: [None if not np.isfinite(x) else float(x) for x in c['fit'][key]] for key in ('CL', 'CT', 'Dpsi', 'Dphi')}
        band = (1 / c['k']) >= floor
        info = {
            'name': name, 'label': label, 'source': meta.get('source', ''), 't0': meta['t0'], 'hours': meta['nFrames'],
            'grid': f"{meta['nx']}×{meta['ny']} @ {meta['dxKm']:g} km", 'search_km': r, 'floor_km': floor,
            'segment_km': seg * meta['dxKm'], 'segments': c['nseg'],
            'fit_wavelengths_km': [round(1 / x, 1) for x in c['k'][band]],
            'subtidal_frac': res['subtidal_frac'], 'rot_share': c['rot_frac'], 'fit': fit, 'validation': val,
            'raw_hours': res['raw_hours'], 'valid_hours': res['valid_hours'],
            'figures': {'footprint': f'analysis/ncei/{foot}', 'spectra': f'analysis/ncei/{spec}'},
        }
        networks.append(info)
        ps = fit['Dpsi']
        print(f"{label}: segs {c['nseg']}, fit band {info['fit_wavelengths_km'][-1] if info['fit_wavelengths_km'] else '–'}–"
              f"{info['fit_wavelengths_km'][0] if info['fit_wavelengths_km'] else '–'} km ({band.sum()} bins) | "
              f"D_ψ {ps[0]:+.2f} ± {ps[1] or float('nan'):.2f} | C_L {fit['CL'][0]:+.2f} C_T {fit['CT'][0]:+.2f} | "
              f"rot share {c['rot_frac']:.2f} | subtidal {res['subtidal_frac']:.0%} | "
              f"validation: −5/3 bias {val['-1.67']['bias']:+.2f}±{val['-1.67']['scatter']:.2f}, "
              f"−3 bias {val['-3.00']['bias']:+.2f}±{val['-3.00']['scatter']:.2f}, share RMS {val['share_rms']:.2f}")

    gistda = None
    gpath = os.path.join(ROOT, 'analysis', 'report', 'summary.json')
    if os.path.exists(gpath):
        gistda = json.load(open(gpath))
    rot = None
    rpath = os.path.join(ROOT, 'analysis', 'rotation', 'rotation_test.json')
    if os.path.exists(rpath):
        rot = json.load(open(rpath))
    lat = None
    lpath = os.path.join(ROOT, 'analysis', 'latitude', 'latitude_test.json')
    if os.path.exists(lpath):
        lat = json.load(open(lpath))
    with open(a.web, 'w') as fh:
        fh.write('/* generated by tools/hfr_ncei_report.py — data for analysis.html */\n')
        fh.write('window.ANALYSIS = ' + json.dumps({'ncei': networks, 'gistda': gistda, 'rotation': rot, 'latitude': lat},
                                                   indent=1) + ';\n')
    print(f'wrote {a.web} and figures in {a.out_dir}')


if __name__ == '__main__':
    main()
