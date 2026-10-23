#!/usr/bin/env python3
"""hfr_ncei.py — stream CODAR radial files from the NOAA NCEI archive and combine them into
hourly total-vector fields, without storing any raw files.

NCEI serves the IOOS HF radar radials (LLUV `.ruv` text files) per site and month, e.g.
  https://www.ncei.noaa.gov/data/oceans/ioos/hfradar/radial/2020/202012/CODAR/<SITE>/
This tool lists each site directory once, fetches the hourly files in parallel straight into
memory, keeps only (lon, lat, radial speed, heading) of QC-passed radials, and combines them:

  each radial i gives one equation  u·sin(H_i) + v·cos(H_i) = VELO_i   (CODAR: VELU = VELO·sin H)
  → unweighted least squares for (u, v) at every grid node from all radials within --search-km,
    accepted if ≥ --min-radials radials from ≥ 2 sites, GDOP ≤ 2/sin(--min-angle) and a radial lies
    within --max-nearest-km (keeps vectors off land and out of data holes).

Outputs (same formats as hfr_preprocess.py, on a local Cartesian grid with exact spacing):
  data/<name>.js, analysis/npz/<name>.npz, and the viewer manifest is refreshed.

Example (central California long-range + standard-range SeaSondes, December 2020):
  python tools/hfr_ncei.py https://www.ncei.noaa.gov/data/oceans/ioos/hfradar/radial/2020/202012/CODAR/ \\
      --sites BIGC,PILR,PSLR,FORT,GCYN,MONT,PPNS --res-km 6 --search-km 9 --name hfr_ncei_ccal_202012
"""
import argparse
import math
import os
import re
import sys
import time
import urllib.request
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hfr_preprocess  # noqa: E402  (export format)
from process_all_hfr import write_manifest  # noqa: E402

FNAME = re.compile(r'RDL_([im])_CODAR_(\w+?)_(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})\.ruv')


def http_get(url, tries=4, timeout=90):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'hfr_ncei/1.0'}),
                                        timeout=timeout) as r:
                return r.read().decode('latin-1')
        except Exception as e:                                  # noqa: BLE001 — retry any network error
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def list_site(base, site):
    """{hour epoch: url} for one site; prefers measured-pattern (RDL_m) over ideal (RDL_i) files."""
    html = http_get(base.rstrip('/') + f'/{site}/')
    best = {}
    for m in FNAME.finditer(html):
        kind, _, y, mo, d, hh, mi = m.groups()
        if mi != '00':
            continue                                            # 30-min sites: keep the on-the-hour file
        t = datetime(int(y), int(mo), int(d), int(hh), tzinfo=timezone.utc).timestamp()
        if t not in best or (kind == 'm' and best[t][0] == 'i'):
            best[t] = (kind, base.rstrip('/') + f'/{site}/{m.group(0)}')
    return {t: url for t, (_, url) in best.items()}


def parse_ruv(text, max_speed_cms):
    """First LLUV table → (lon, lat, VELO [m/s], HEAD [rad]) of QC-passed radials, plus header info."""
    cols, rows, hdr = None, [], {}
    for line in text.splitlines():
        if line.startswith('%'):
            if line.startswith('%TableColumnTypes:') and cols is None:
                cols = line.split(':', 1)[1].split()
            elif line.startswith('%TableEnd') and cols is not None and rows:
                break
            elif ':' in line and cols is None:
                k, v = line[1:].split(':', 1)
                hdr[k.strip()] = v.strip()
            continue
        if cols is not None and line.strip():
            rows.append(line.split())
    if not cols or not rows:
        return None, hdr
    a = np.array([r[:len(cols)] for r in rows if len(r) >= len(cols)], dtype=float)
    c = {name: i for i, name in enumerate(cols)}
    ok = np.abs(a[:, c['VELO']]) <= max_speed_cms
    if 'VFLG' in c:
        ok &= a[:, c['VFLG']] == 0
    a = a[ok]
    return (a[:, c['LOND']], a[:, c['LATD']], a[:, c['VELO']] / 100.0, np.radians(a[:, c['HEAD']])), hdr


def combine(points_km, sinH, cosH, vel, site, grid_km, a, nsites):
    """Vectorised unweighted least-squares totals at all grid nodes for one hour."""
    tree = cKDTree(points_km)
    nbrs = tree.query_ball_point(grid_km, a.search_km)
    node = np.repeat(np.arange(len(grid_km)), [len(n) for n in nbrs])
    idx = np.concatenate([np.asarray(n, int) for n in nbrs]) if len(node) else np.zeros(0, int)
    G = len(grid_km)
    acc = lambda w: np.bincount(node, weights=w, minlength=G)
    s, c, v = sinH[idx], cosH[idx], vel[idx]
    n = np.bincount(node, minlength=G).astype(float)
    Sss, Scc, Ssc, Ssv, Scv = acc(s * s), acc(c * c), acc(s * c), acc(s * v), acc(c * v)
    has_site = np.bincount(node * nsites + site[idx], minlength=G * nsites).reshape(G, nsites) > 0
    nsite = has_site.sum(axis=1)
    det = Sss * Scc - Ssc ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        u = (Scc * Ssv - Ssc * Scv) / det
        w = (Sss * Scv - Ssc * Ssv) / det
        gdop = np.sqrt(n * (Sss + Scc) / det)                   # = 2/sin(θ) for two crossing directions
    nearest, _ = tree.query(grid_km)
    ok = (n >= a.min_radials) & (nsite >= 2) & (gdop <= 2 / math.sin(math.radians(a.min_angle))) \
        & (nearest <= a.max_nearest_km) & (np.hypot(u, w) <= a.max_speed_cms / 100.0)
    u[~ok] = np.nan
    w[~ok] = np.nan
    return u, w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('base', help='NCEI month directory, e.g. …/hfradar/radial/2020/202012/CODAR/')
    ap.add_argument('--sites', required=True, help='comma-separated site codes')
    ap.add_argument('--res-km', type=float, default=6.0, help='total-vector grid spacing (default 6 km)')
    ap.add_argument('--search-km', type=float, default=9.0, help='radial search radius (default 9 km)')
    ap.add_argument('--min-radials', type=int, default=3)
    ap.add_argument('--min-angle', type=float, default=30.0, help='minimum crossing angle via GDOP (deg)')
    ap.add_argument('--max-nearest-km', type=float, default=None, help='default: 0.75 × search radius')
    ap.add_argument('--max-speed-cms', type=float, default=150.0)
    ap.add_argument('--bbox', type=float, nargs=4, metavar=('LON0', 'LON1', 'LAT0', 'LAT1'),
                    help='grid bounds (default: extent of the radials)')
    ap.add_argument('--start', help='UTC start (default: whole directory)')
    ap.add_argument('--end', help='UTC end, exclusive')
    ap.add_argument('--threads', type=int, default=24)
    ap.add_argument('--name', required=True, help='output name, e.g. hfr_ncei_ccal_202012')
    ap.add_argument('--data-dir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--npz-dir', default=os.path.join(ROOT, 'analysis', 'npz'))
    a = ap.parse_args()
    a.max_nearest_km = a.max_nearest_km or 0.75 * a.search_km
    sites = [s.strip() for s in a.sites.split(',') if s.strip()]

    # 1) Directory listings → one URL per site-hour.
    t_start = time.time()
    with ThreadPoolExecutor(len(sites)) as ex:
        listings = dict(zip(sites, ex.map(lambda s: list_site(a.base, s), sites)))
    t_lo = hfr_preprocess.parse_time(a.start, 0) if a.start else -math.inf
    t_hi = hfr_preprocess.parse_time(a.end, 0) if a.end else math.inf
    listings = {s: {t: u for t, u in l.items() if t_lo <= t < t_hi} for s, l in listings.items()}
    hours = sorted(set().union(*[set(l) for l in listings.values()]))
    for s in sites:
        print(f'  {s}: {len(listings[s])} hourly files')
    jobs = [(s, t, url) for s in sites for t, url in listings[s].items()]
    print(f'{len(jobs)} files to stream for {len(hours)} hours ({len(sites)} sites)')

    # 2) Stream and parse in parallel; keep only compact arrays.
    radials, origins, freq, nbytes = {}, {}, {}, 0
    def work(job):
        s, t, url = job
        text = http_get(url)
        data, hdr = parse_ruv(text, a.max_speed_cms)
        return s, t, data, hdr, len(text)
    done = 0
    with ThreadPoolExecutor(a.threads) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            try:
                s, t, data, hdr, n = f.result()
            except Exception as e:                              # noqa: BLE001
                print(f'  skipped a file: {e}', file=sys.stderr)
                continue
            nbytes += n
            done += 1
            if data is not None and len(data[0]):
                radials[(s, t)] = data
            if s not in origins and 'Origin' in hdr:
                origins[s] = [float(x) for x in hdr['Origin'].split()[:2]]
                freq[s] = hdr.get('TransmitCenterFreqMHz', '?')
            if done % 500 == 0:
                print(f'  … {done}/{len(jobs)} files, {nbytes / 1e6:.0f} MB streamed', file=sys.stderr)
    print(f'streamed {done} files ({nbytes / 1e6:.0f} MB) in {time.time() - t_start:.0f} s; nothing written to disk')

    # 3) Grid on a local Cartesian frame centred on the network.
    all_lon = np.concatenate([d[0] for d in radials.values()])
    all_lat = np.concatenate([d[1] for d in radials.values()])
    if a.bbox:
        lon0, lon1, lat0, lat1 = a.bbox
    else:
        lon0, lon1 = np.percentile(all_lon, [0.2, 99.8])
        lat0, lat1 = np.percentile(all_lat, [0.2, 99.8])
    latc, lonc = 0.5 * (lat0 + lat1), 0.5 * (lon0 + lon1)
    kx, ky = 111.32 * math.cos(math.radians(latc)), 110.57
    xs = np.arange((lon0 - lonc) * kx, (lon1 - lonc) * kx + 1e-9, a.res_km)
    ys = np.arange((lat0 - latc) * ky, (lat1 - latc) * ky + 1e-9, a.res_km)
    GX, GY = np.meshgrid(xs, ys)
    grid_km = np.column_stack([GX.ravel(), GY.ravel()])
    glon, glat = lonc + xs / kx, latc + ys / ky
    site_id = {s: i for i, s in enumerate(sites)}

    t0, step = hours[0], 3600.0
    nt = int(round((hours[-1] - t0) / step)) + 1
    U = np.full((nt, len(ys), len(xs)), np.nan, np.float32)
    V = np.full_like(U, np.nan)
    for t in hours:
        parts = [(radials[(s, t)], site_id[s]) for s in sites if (s, t) in radials]
        if len({sid for _, sid in parts}) < 2:
            continue
        lon = np.concatenate([p[0][0] for p in parts]); lat = np.concatenate([p[0][1] for p in parts])
        vel = np.concatenate([p[0][2] for p in parts]); head = np.concatenate([p[0][3] for p in parts])
        sid = np.concatenate([np.full(len(p[0][0]), p[1]) for p in parts])
        pts = np.column_stack([(lon - lonc) * kx, (lat - latc) * ky])
        u, v = combine(pts, np.sin(head), np.cos(head), vel, sid, grid_km, a, len(sites))
        k = int(round((t - t0) / step))
        U[k], V[k] = u.reshape(GX.shape), v.reshape(GX.shape)
    cover = np.isfinite(U).mean(axis=(1, 2))
    print(f'grid {len(xs)}×{len(ys)} at {a.res_km:g} km ({glon[0]:.2f}…{glon[-1]:.2f}°E, {glat[0]:.2f}…{glat[-1]:.2f}°N), '
          f'{nt} frames, node coverage min/mean/max {cover.min():.0%}/{cover.mean():.0%}/{cover.max():.0%}')
    print('sites: ' + ', '.join(f'{s} {freq.get(s, "?")[:5]} MHz' for s in sites))

    # 4) Export in the standard formats and refresh the viewer manifest.
    os.makedirs(a.data_dir, exist_ok=True)
    os.makedirs(a.npz_dir, exist_ok=True)
    ns = Namespace(out=os.path.join(a.data_dir, a.name), npz=os.path.join(a.npz_dir, a.name + '.npz'),
                   step_hours=1.0, sql=None,
                   table=f'NOAA NCEI CODAR radials ({",".join(sites)}), UWLS totals r={a.search_km:g} km')
    hfr_preprocess.export(ns, U, V, glon, glat, float(glon[1] - glon[0]), float(glat[1] - glat[0]), a.res_km, t0, step, nt)
    write_manifest(a.data_dir)


if __name__ == '__main__':
    main()
