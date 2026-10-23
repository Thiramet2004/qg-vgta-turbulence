#!/usr/bin/env python3
"""hfr_report.py — aggregated subtidal rotational/divergent spectra for all processed months,
with wet (SW monsoon) vs dry (NE monsoon + pre-monsoon) seasonal comparison.

Runs hfr_spectra.analyze() on every analysis/npz/hfr_YYYYMM.npz (in parallel), then:
  * per-month table: data volume, subtidal share, rotational share, fitted slopes (λ ≥ 20 km)
  * seasonal mean spectra with 95 % bootstrap intervals (resampling months)
  * bootstrap slope and rotational-share estimates per season and their wet − dry difference
  * figures + a Markdown report in analysis/report/

Example:
  python tools/hfr_report.py --jobs 6
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
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hfr_spectra as S  # noqa: E402

SEASONS = {'wet': (5, 6, 7, 8, 9, 10), 'dry': (11, 12, 1, 2, 3, 4)}   # SW monsoon | NE monsoon + pre-monsoon
COLORS = {'wet': '#1f77b4', 'dry': '#d62728'}


def season_of(ym):
    m = int(ym[4:])
    return next(s for s, months in SEASONS.items() if m in months)


def run(args):
    path, kw = args
    warnings.simplefilter('ignore', RuntimeWarning)
    r = S.analyze(path, **kw)
    d = np.load(path)
    r['coverage'] = float(np.isfinite(d['u']).mean())
    return r


def bootstrap(stack, fn, n=4000, seed=0):
    """Percentile 95 % interval of fn(mean over resampled months)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(stack), (n, len(stack)))
    vals = np.array([fn(stack[i].mean(axis=0)) for i in idx])
    return np.nanpercentile(vals, [2.5, 50, 97.5], axis=0), vals


def main():
    warnings.simplefilter('ignore', RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz-dir', default=os.path.join(ROOT, 'analysis', 'npz'))
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'analysis', 'report'))
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--direction', default='combined', choices=['combined', 'zonal', 'meridional'],
                    help='transects used for the seasonal spectra (default: both directions averaged)')
    ap.add_argument('--segment', type=int, default=26,
                    help='segment length in nodes for both directions (default 26 = 52 km: the longest that fits '
                         'the post-2025-11 footprint)')
    ap.add_argument('--config-split', default='202511',
                    help='first month of the second radar configuration (footprint change); seasons are also '
                         'compared within each configuration')
    ap.add_argument('--min-segments', type=int, default=200, help='exclude months with fewer segments')
    ap.add_argument('--min-hours', type=int, default=300, help='exclude months with fewer filtered hours')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    kw = dict(lowpass_kind='lanczos', cutoff_h=72.0, half_len_h=120.0, segment=a.segment,
              segment_meridional=a.segment, max_fill=2, min_wl_km=20.0)

    files = sorted(f for f in glob.glob(os.path.join(a.npz_dir, 'hfr_*.npz'))
                   if re.fullmatch(r'hfr_\d{6}\.npz', os.path.basename(f)))      # GISTDA monthly files only
    months = [re.search(r'(\d{6})', os.path.basename(f)).group(1) for f in files]
    with ProcessPoolExecutor(a.jobs) as ex:
        results = dict(zip(months, ex.map(run, [(f, kw) for f in files])))

    # ── per-month table ──
    rows = []
    for ym, r in results.items():
        z, m = r['spectra'].get(a.direction), r['spectra'].get('meridional')
        ok = bool(z) and z['nseg'] >= a.min_segments and r['valid_hours'] >= a.min_hours
        rows.append({
            'month': f'{ym[:4]}-{ym[4:]}', 'ym': ym, 'season': season_of(ym), 'used': ok,
            'config': 'A' if ym < a.config_split else 'B',
            'raw_hours': r['raw_hours'], 'filtered_hours': r['valid_hours'], 'coverage': r['coverage'],
            'subtidal_frac': r['subtidal_frac'], 'nseg': z['nseg'] if z else 0,
            'slope_psi': z['fit']['Dpsi'] if z else (math.nan,) * 3,
            'slope_phi': z['fit']['Dphi'] if z else (math.nan,) * 3,
            'rot_frac': z['rot_frac'] if z else math.nan,
            'rot_frac_merid': m['rot_frac'] if m else math.nan,
        })
    used = [r for r in rows if r['used']]
    k = next(results[r['ym']]['spectra'][a.direction]['k'] for r in used)
    band = (1 / k) >= kw['min_wl_km']

    # ── seasonal aggregation with bootstrap over months ──
    def aggregate(ms):
        sp = {key: np.array([results[x]['spectra'][a.direction][key] for x in ms]) for key in ('CL', 'CT', 'Dpsi', 'Dphi')}
        slope = lambda E: S.fit_power_law(k, E, kw['min_wl_km'])[0]
        share = lambda pair: np.sum(pair[0][band]) / np.sum(pair[0][band] + pair[1][band])
        ci_psi, _ = bootstrap(sp['Dpsi'], lambda E: E)
        ci_phi, _ = bootstrap(sp['Dphi'], lambda E: E)
        sl_psi, boot_psi = bootstrap(sp['Dpsi'], slope, seed=1)
        sl_phi, _ = bootstrap(sp['Dphi'], slope, seed=2)
        both = np.stack([sp['Dpsi'], sp['Dphi']], axis=1)
        sh, boot_sh = bootstrap(both, share, seed=3)
        dk = k[1] - k[0]
        energy = lambda E: np.sum(E[band]) * dk                      # KE in the λ ≥ min_wl band [(m/s)²]
        en, boot_en = bootstrap(sp['Dpsi'], energy, seed=4)
        return dict(months=ms, n=len(ms), mean={key: v.mean(axis=0) for key, v in sp.items()},
                                ci_psi=ci_psi, ci_phi=ci_phi, slope_psi=sl_psi, slope_phi=sl_phi,
                                share=sh, boot_psi=boot_psi, boot_share=boot_sh, energy=en, boot_energy=boot_en,
                                energy_mean=energy(sp['Dpsi'].mean(axis=0)),
                                slope_psi_mean=slope(sp['Dpsi'].mean(axis=0)),
                                slope_phi_mean=slope(sp['Dphi'].mean(axis=0)),
                                share_mean=share((sp['Dpsi'].mean(axis=0), sp['Dphi'].mean(axis=0))),
                                subtidal=np.mean([results[x]['subtidal_frac'] for x in ms]))

    seasonal, by_config = {}, {}
    for season in SEASONS:
        ms = [r['ym'] for r in used if r['season'] == season]
        if len(ms) >= 2:
            seasonal[season] = aggregate(ms)
        for cfg in ('A', 'B'):
            ms = [r['ym'] for r in used if r['season'] == season and r['config'] == cfg]
            if len(ms) >= 2:
                by_config[(cfg, season)] = aggregate(ms)
    diff = None
    if {'wet', 'dry'} <= seasonal.keys():
        n = min(len(seasonal['wet']['boot_psi']), len(seasonal['dry']['boot_psi']))
        d_sl = seasonal['wet']['boot_psi'][:n] - seasonal['dry']['boot_psi'][:n]
        d_sh = seasonal['wet']['boot_share'][:n] - seasonal['dry']['boot_share'][:n]
        r_en = seasonal['wet']['boot_energy'][:n] / seasonal['dry']['boot_energy'][:n]
        diff = {'slope': np.nanpercentile(d_sl, [2.5, 50, 97.5]), 'share': np.nanpercentile(d_sh, [2.5, 50, 97.5]),
                'energy_ratio': np.nanpercentile(r_en, [2.5, 50, 97.5])}

    # ── figures ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, key, ci_key, title in ((axes[0], 'Dpsi', 'ci_psi', r'Rotational $D_\psi(k)$'),
                                   (axes[1], 'Dphi', 'ci_phi', r'Divergent $D_\phi(k)$')):
        floor = 1e-6
        for season, s in seasonal.items():
            ci, E = s[ci_key], s['mean'][key]
            ax.fill_between(k, np.clip(ci[0], floor, None), np.clip(ci[2], floor, None), color=COLORS[season], alpha=0.18, lw=0)
            ax.loglog(k, np.where(E > 0, E, np.nan), 'o-', ms=3.5, color=COLORS[season], label=f"{season} (n={s['n']} months)")
            neg = E <= 0                                              # anisotropy: isotropic split fails
            if neg.any():
                ax.loglog(k[neg], np.maximum(-E[neg], floor), 'o', ms=5, mfc='none', color=COLORS[season],
                          label=f'{season}: |negative estimate|')
        ax.set_ylim(bottom=floor)
        ref = np.nanmax([s['mean'][key][0] for s in seasonal.values()]) * 1.3
        for slope, style in ((-5 / 3, '--'), (-3, ':')):
            ax.loglog(k, ref * (k / k[0]) ** slope, 'k' + style, lw=1, label=f'$k^{{{slope:.2g}}}$')
        ax.axvspan(1 / kw['min_wl_km'], k[-1] * 1.05, color='0.8', alpha=0.45, lw=0)
        ax.set_title(title)
        ax.set_xlabel('wavenumber [cycles/km]')
        ax.grid(True, which='both', alpha=0.25)
        sec = ax.secondary_xaxis('top', functions=(lambda x: 1 / np.maximum(x, 1e-9), lambda x: 1 / np.maximum(x, 1e-9)))
        sec.set_xlabel('wavelength [km]')
    axes[0].set_ylabel('E(k) [(m/s)$^2$ / (cycles/km)]')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Subtidal (Lanczos 72 h) HF radar spectra, upper Gulf of Thailand — {a.direction} transects; '
                 f'grey: λ < {kw["min_wl_km"]:g} km (radar smoothing, not fitted)', fontsize=10)
    fig.tight_layout()
    f1 = os.path.join(a.out_dir, 'seasonal_spectra.png')
    fig.savefig(f1, dpi=150)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    x = np.arange(len(rows))
    for r_i, r in enumerate(rows):
        c = COLORS[r['season']] if r['used'] else '0.6'
        s, se, _ = r['slope_psi']
        axes[0].errorbar(r_i, s, yerr=0 if not np.isfinite(se) else se, fmt='o', color=c, mfc=c if r['used'] else 'none')
        axes[1].plot(r_i, r['rot_frac'], 'o', color=c, mfc=c if r['used'] else 'none')
        axes[2].bar(r_i, r['subtidal_frac'], color=c, alpha=0.8 if r['used'] else 0.3)
    for y, st in ((-5 / 3, '--'), (-3, ':')):
        axes[0].axhline(y, color='k', ls=st, lw=1)
    axes[0].set_ylabel(r'$D_\psi$ slope (λ ≥ 20 km)')
    axes[1].axhline(0.5, color='0.5', lw=0.8)
    axes[1].set_ylabel('rotational share')
    axes[2].set_ylabel('subtidal / total KE')
    axes[2].set_xticks(x, [r['month'] for r in rows], rotation=60, fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].set_title('Monthly diagnostics (blue: wet SW monsoon, red: dry; hollow/grey: excluded for low data)', fontsize=10)
    fig.tight_layout()
    f2 = os.path.join(a.out_dir, 'monthly_diagnostics.png')
    fig.savefig(f2, dpi=150)

    # ── machine-readable outputs ──
    with open(os.path.join(a.out_dir, 'monthly_table.csv'), 'w') as fh:
        fh.write('month,season,config,used,raw_hours,filtered_hours,coverage,subtidal_frac,segments,'
                 'Dpsi_slope,Dpsi_slope_se,Dphi_slope,rot_share,rot_share_meridional\n')
        for r in rows:
            fh.write(f"{r['month']},{r['season']},{r['config']},{int(r['used'])},{r['raw_hours']},{r['filtered_hours']},"
                     f"{r['coverage']:.3f},{r['subtidal_frac']:.3f},{r['nseg']},{r['slope_psi'][0]:.3f},"
                     f"{r['slope_psi'][1]:.3f},{r['slope_phi'][0]:.3f},{r['rot_frac']:.3f},{r['rot_frac_merid']:.3f}\n")
    with open(os.path.join(a.out_dir, 'seasonal_spectra.csv'), 'w') as fh:
        fh.write('season,k_cpkm,wavelength_km,C_L,C_T,D_psi,D_psi_lo95,D_psi_hi95,D_phi,D_phi_lo95,D_phi_hi95\n')
        for season, s in seasonal.items():
            for i, kk in enumerate(k):
                fh.write(f"{season},{kk:.6g},{1 / kk:.4g},{s['mean']['CL'][i]:.6g},{s['mean']['CT'][i]:.6g},"
                         f"{s['mean']['Dpsi'][i]:.6g},{s['ci_psi'][0][i]:.6g},{s['ci_psi'][2][i]:.6g},"
                         f"{s['mean']['Dphi'][i]:.6g},{s['ci_phi'][0][i]:.6g},{s['ci_phi'][2][i]:.6g}\n")

    # ── Markdown report ──
    val_path = os.path.join(a.out_dir, 'validation.json')
    val = json.load(open(val_path)) if os.path.exists(val_path) else None
    L = []
    L.append('# Subtidal rotational spectra of HF radar currents — upper Gulf of Thailand\n')
    L.append(f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `tools/hfr_report.py` from "
             f"{len(rows)} monthly GISTDA HF radar files ({rows[0]['month']} – {rows[-1]['month']})._\n")
    notes = os.path.join(a.out_dir, 'notes.md')
    if os.path.exists(notes):
        L.append(open(notes).read().strip() + '\n')
    followup = os.path.join(a.out_dir, 'qnse_gistda.md')        # written by tools/hfr_qnse_gistda.py
    if os.path.exists(followup):
        L.append(open(followup).read().strip() + '\n')
    L.append('## Method\n')
    L.append('* Native 2 km total-vector grid (`--native-xy`), 12.7–13.5°N, 100.2–100.9°E; anomalies about each month’s mean.')
    L.append('* Lanczos low-pass, 72 h half-power, ±120 h window: gain 0.000 at 12.42 h and 24 h, 0.016 at the '
             '53.3 h inertial period (99.97 % of near-inertial energy removed), ≥ 0.9 beyond 96 h.')
    wl_fit = ', '.join(f'{1 / kk:.0f}' for kk in k[band])
    L.append(f"* 1-D spectra from gap-tolerant {a.segment}-node ({a.segment * 2} km) segments (interior gaps ≤ 2 nodes "
             f"filled), de-meaned, Tukey(0.5) taper; **{a.direction}** = zonal and meridional transects averaged "
             '(segment-weighted, isotropy assumed). The segment length is the longest that fits the post-2025-11 '
             'footprint, so all months are analysed identically.')
    L.append('* Rotational/divergent split of the 1-D spectra after Bühler, Callies & Ferrari (2014).')
    L.append(f"* Power laws fitted only for λ ≥ {kw['min_wl_km']:g} km — {band.sum()} bins ({wl_fit} km), i.e. a "
             'factor-of-two range in wavelength; shorter scales are shaped by the least-squares combination of '
             'radials into totals.')
    L.append(f'* Seasons: wet = SW monsoon (May–Oct), dry = NE monsoon + pre-monsoon (Nov–Apr). Months with < '
             f'{a.min_segments} segments or < {a.min_hours} filtered hours are excluded. 95 % intervals: bootstrap '
             'over months (4000 resamples).\n')
    if val:
        s53, s3, sh = val['slope_-1.67'], val['slope_-3.00'], val['rot_share']
        L.append('### Method validation (known-answer synthetic months, real gap mask)\n')
        L.append(f"* Single-month slope for a true k^−5/3: bias {s53['bias']:+.2f}, scatter ±{s53['scatter']:.2f}; "
                 f"for a true k^−3: bias {s3['bias']:+.2f}, scatter ±{s3['scatter']:.2f}.")
        L.append(f"* Rotational share: bias {sh['bias']:+.2f}, RMS error {sh['rms_error']:.2f}.")
        L.append('* Averaging ~9 months per season reduces the random part by ~3×, so −5/3 and −3 are separable '
                 'at the seasonal level, but single-month slopes are not conclusive.\n')
    L.append('## Seasonal results\n')
    L.append('| Group | Months | Subtidal share of KE | Rotational KE, λ ≥ 20 km [cm²/s²] (95 % CI) | D_ψ slope (95 % CI) | D_φ slope (95 % CI) | Rotational share, λ ≥ 20 km (95 % CI) |')
    L.append('|---|---|---|---|---|---|---|')
    groups = [(season, s) for season, s in seasonal.items()]
    split = f'{a.config_split[:4]}-{a.config_split[4:]}'
    groups += [(f'{season}, config {cfg} ({"before" if cfg == "A" else "from"} {split})', s)
               for (cfg, season), s in sorted(by_config.items())]
    def ci_txt(mean, ci, fmt='+.2f'):
        if not np.isfinite(mean):
            return 'n/a (D ≤ 0 in fit band)'
        return f'{mean:{fmt}} ({ci[0]:{fmt}} … {ci[2]:{fmt}})'

    for label, s in groups:
        share_txt = ci_txt(s['share_mean'], s['share'], '.2f') + (' ⚠ >1: anisotropic' if s['share_mean'] > 1 else '')
        L.append(f"| {label} | {s['n']} | {s['subtidal']:.0%} | {ci_txt(s['energy_mean'] * 1e4, s['energy'] * 1e4, '.1f')} | "
                 f"{ci_txt(s['slope_psi_mean'], s['slope_psi'])} | {ci_txt(s['slope_phi_mean'], s['slope_phi'])} | {share_txt} |")
    L.append('\nConfiguration A (Feb–Oct 2025) and B (Nov 2025 onward) sample different footprints (B is narrower '
             'E–W and extends further south), so the within-configuration rows are the cleaner seasonal comparison.')
    if diff:
        L.append(f"\nWet / dry rotational KE (λ ≥ 20 km): ×{diff['energy_ratio'][1]:.2f} (95 % CI "
                 f"×{diff['energy_ratio'][0]:.2f} … ×{diff['energy_ratio'][2]:.2f}).")
        L.append(f"Wet − dry: D_ψ slope {diff['slope'][1]:+.2f} (95 % CI {diff['slope'][0]:+.2f} … "
                 f"{diff['slope'][2]:+.2f}); rotational share {diff['share'][1]:+.2f} "
                 f"({diff['share'][0]:+.2f} … {diff['share'][2]:+.2f}).\n")
    L.append('![Seasonal spectra](seasonal_spectra.png)\n')
    L.append('## Monthly diagnostics\n')
    L.append('| Month | Season | Config | Used | Hours raw / filtered | Node coverage | Subtidal KE share | Segments | D_ψ slope (2 bins) | Rotational share |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        L.append(f"| {r['month']} | {r['season']} | {r['config']} | {'yes' if r['used'] else '**no**'} | "
                 f"{r['raw_hours']} / {r['filtered_hours']} | {r['coverage']:.0%} | {r['subtidal_frac']:.0%} | "
                 f"{r['nseg']} | {r['slope_psi'][0]:+.2f} | {r['rot_frac']:.2f} |")
    L.append('\n![Monthly diagnostics](monthly_diagnostics.png)\n')
    L.append('Data files: `monthly_table.csv`, `seasonal_spectra.csv`.\n')
    with open(os.path.join(a.out_dir, 'hfr_report.md'), 'w') as fh:
        fh.write('\n'.join(L))

    # machine-readable summary for analysis.html
    fin = lambda x: None if x is None or not np.isfinite(x) else float(x)
    summary = {
        'months': [{k2: (fin(v2) if isinstance(v2, float) else v2) for k2, v2 in r.items() if k2 not in ('slope_psi', 'slope_phi')}
                   | {'slope_psi': fin(r['slope_psi'][0]), 'rot_frac': fin(r['rot_frac'])} for r in rows],
        'groups': [{'label': label, 'n': s['n'], 'subtidal': fin(s['subtidal']),
                    'energy_cm2s2': [fin(s['energy_mean'] * 1e4), fin(s['energy'][0] * 1e4), fin(s['energy'][2] * 1e4)],
                    'slope_psi': [fin(s['slope_psi_mean']), fin(s['slope_psi'][0]), fin(s['slope_psi'][2])],
                    'slope_phi': [fin(s['slope_phi_mean']), fin(s['slope_phi'][0]), fin(s['slope_phi'][2])],
                    'rot_share': [fin(s['share_mean']), fin(s['share'][0]), fin(s['share'][2])]} for label, s in groups],
        'diff': {key: [fin(x) for x in v] for key, v in diff.items()} if diff else None,
        'figures': {'seasonal': 'analysis/report/seasonal_spectra.png', 'monthly': 'analysis/report/monthly_diagnostics.png'},
        'validation': val, 'segment_km': a.segment * 2, 'fit_wavelengths_km': [round(1 / kk, 1) for kk in k[band]],
    }
    with open(os.path.join(a.out_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)

    # console summary
    if diff:
        print(f"wet/dry rotational KE ratio ×{diff['energy_ratio'][1]:.2f} [{diff['energy_ratio'][0]:.2f}, {diff['energy_ratio'][2]:.2f}]")
    for season, s in seasonal.items():
        print(f"{season}: n={s['n']} KE_rot {s['energy_mean'] * 1e4:.1f} cm²/s² subtidal {s['subtidal']:.0%} | D_ψ slope {s['slope_psi_mean']:+.2f} "
              f"[{s['slope_psi'][0]:+.2f}, {s['slope_psi'][2]:+.2f}] | D_φ slope {s['slope_phi_mean']:+.2f} "
              f"[{s['slope_phi'][0]:+.2f}, {s['slope_phi'][2]:+.2f}] | rot share {s['share_mean']:.2f} "
              f"[{s['share'][0]:.2f}, {s['share'][2]:.2f}]")
    for (cfg, season), s in sorted(by_config.items()):
        print(f"  config {cfg} {season}: n={s['n']} D_ψ slope {s['slope_psi_mean']:+.2f} "
              f"[{s['slope_psi'][0]:+.2f}, {s['slope_psi'][2]:+.2f}] | rot share {s['share_mean']:.2f} "
              f"[{s['share'][0]:.2f}, {s['share'][2]:.2f}]")
    if diff:
        print(f"wet − dry: slope {diff['slope'][1]:+.2f} [{diff['slope'][0]:+.2f}, {diff['slope'][2]:+.2f}], "
              f"share {diff['share'][1]:+.2f} [{diff['share'][0]:+.2f}, {diff['share'][2]:+.2f}]")
    print('excluded:', ', '.join(r['month'] for r in rows if not r['used']) or 'none')
    print(f'wrote {a.out_dir}/hfr_report.md, seasonal_spectra.png, monthly_diagnostics.png, *.csv')


if __name__ == '__main__':
    main()
