#!/usr/bin/env python3
"""hfr_rtv.py — stream a month of official HFRNet total-vector products (RTV, NetCDF4) from the
NOAA NCEI archive and cut them into analysis areas, with no raw files stored.

  https://www.ncei.noaa.gov/data/oceans/ioos/hfradar/rtv/YYYY/YYYYMM/<REGION>/
      YYYYMMDDHH00_hfr_<region>_<res>_rtv_uwls_SIO.nc     (hourly, unweighted least squares)

Regions in the archive include USWC, USEGC, USHI, PRVI, GAK, AKNS. These are total vectors like
CODAR .tuv files (HFRNet: 10 km search radius, ≥ 2 sites, ≥ 3 radials, HDOP ≤ 1.25 for 6 km).

Only valid cells are kept while streaming. Afterwards, "areas" are found automatically: connected
clusters of cells with data more than --min-avail of the hours, at least --min-km across; clusters
spanning more than --max-lat-span degrees are split into latitude bands (so f is ~constant).
Each area is written as analysis/npz/rtv_<region>_<YYYYMM>_<lat>N_<lon>.npz in the standard format.

Example:
  python tools/hfr_rtv.py 2020/202012 --regions USWC,USEGC,USHI,PRVI,GAK,AKNS --res 6km
"""
import argparse
import json
import math
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = 'https://www.ncei.noaa.gov/data/oceans/ioos/hfradar/rtv'
R_KM = 6371.0


def http_get(url, tries=4, timeout=120, binary=False):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'hfr_rtv/1.0'}),
                                        timeout=timeout) as r:
                data = r.read()
                return data if binary else data.decode('latin-1')
        except Exception:                                        # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def list_month(month, region, res, source):
    html = http_get(f'{BASE}/{month}/{region}/')
    pat = re.compile(rf'(\d{{12}})_hfr_{region.lower()}_{res}_rtv_uwls_{source}\.nc')
    files = {}
    for m in pat.finditer(html):
        t = datetime.strptime(m.group(1), '%Y%m%d%H%M').replace(tzinfo=timezone.utc).timestamp()
        files[t] = f'{BASE}/{month}/{region}/{m.group(0)}'
    return files


def read_rtv(blob):
    import netCDF4
    d = netCDF4.Dataset('mem', memory=blob)
    u = d['u'][0]; v = d['v'][0]
    ok = ~(np.ma.getmaskarray(u) | np.ma.getmaskarray(v))
    idx = np.flatnonzero(ok)
    out = (idx.astype(np.int32), np.asarray(u).ravel()[idx].astype(np.float32), np.asarray(v).ravel()[idx].astype(np.float32))
    grid = (np.asarray(d['lat'][:]), np.asarray(d['lon'][:]))
    d.close()
    return out, grid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('month', help='YYYY/YYYYMM, e.g. 2020/202012')
    ap.add_argument('--regions', default='USWC,USEGC,USHI,PRVI,GAK,AKNS')
    ap.add_argument('--res', default='6km', help='1km, 2km or 6km (default 6km: largest footprints)')
    ap.add_argument('--source', default='SIO', help='SIO or NDBC copy of the product')
    ap.add_argument('--min-avail', type=float, default=0.5, help='cell kept if data in ≥ this fraction of hours')
    ap.add_argument('--min-km', type=float, default=80.0, help='minimum area extent (both directions)')
    ap.add_argument('--max-lat-span', type=float, default=3.0)
    ap.add_argument('--threads', type=int, default=24)
    ap.add_argument('--npz-dir', default=os.path.join(ROOT, 'analysis', 'npz'))
    a = ap.parse_args()
    os.makedirs(a.npz_dir, exist_ok=True)
    ym = a.month.split('/')[-1]
    written = []

    for region in [r.strip() for r in a.regions.split(',') if r.strip()]:
        t0 = time.time()
        try:
            files = list_month(a.month, region, a.res, a.source)
        except Exception as e:                                   # noqa: BLE001 — e.g. 404: region not operating
            print(f'{region}: no data for {a.month} ({e})'); continue
        if not files:
            print(f'{region}: no {a.res} files'); continue
        hours = sorted(files)
        tstart, nt = hours[0], int(round((hours[-1] - hours[0]) / 3600)) + 1
        frames, grid, nbytes = {}, None, 0
        with ThreadPoolExecutor(a.threads) as ex:
            futs = {ex.submit(http_get, url, binary=True): t for t, url in files.items()}
            for f in as_completed(futs):
                try:
                    blob = f.result()
                except Exception as e:                           # noqa: BLE001
                    print(f'  {region}: skipped a file ({e})', file=sys.stderr); continue
                nbytes += len(blob)
                (idx, u, v), g = read_rtv(blob)
                grid = grid or g
                frames[int(round((futs[f] - tstart) / 3600))] = (idx, u, v)
        lat, lon = grid
        ny, nx = len(lat), len(lon)
        count = np.zeros(ny * nx, np.int32)
        for idx, _, _ in frames.values():
            count[idx] += 1
        avail = (count / nt).reshape(ny, nx)
        print(f'{region}: {len(frames)} hourly files, {nbytes / 1e6:.0f} MB streamed in {time.time() - t0:.0f} s; '
              f'grid {nx}×{ny}; cells with data > {a.min_avail:.0%} of hours: {(avail > a.min_avail).sum()}')

        # Areas: connected clusters of well-observed cells, split into latitude bands.
        lab, nlab = ndimage.label(avail > a.min_avail)
        dlat = float(np.mean(np.diff(lat))); dlon = float(np.mean(np.diff(lon)))
        for L in range(1, nlab + 1):
            ys, xs = np.where(lab == L)
            band = np.floor((lat[ys] - lat[ys].min()) / a.max_lat_span).astype(int)
            for b in np.unique(band):
                sel = band == b
                if sel.sum() < 20:
                    continue
                y0, y1, x0, x1 = ys[sel].min(), ys[sel].max(), xs[sel].min(), xs[sel].max()
                latc = float(lat[(y0 + y1) // 2])
                ext_y = (y1 - y0 + 1) * dlat * math.pi / 180 * R_KM
                ext_x = (x1 - x0 + 1) * dlon * math.pi / 180 * R_KM * math.cos(math.radians(latc))
                if min(ext_x, ext_y) < a.min_km:
                    continue
                U = np.full((nt, y1 - y0 + 1, x1 - x0 + 1), np.nan, np.float32)
                V = np.full_like(U, np.nan)
                box = (lab[y0:y1 + 1, x0:x1 + 1] == L)
                for k, (idx, u, v) in frames.items():
                    yy, xx = np.unravel_index(idx, (ny, nx))
                    m = (yy >= y0) & (yy <= y1) & (xx >= x0) & (xx <= x1)
                    U[k, yy[m] - y0, xx[m] - x0] = u[m]
                    V[k, yy[m] - y0, xx[m] - x0] = v[m]
                U[:, ~box] = np.nan; V[:, ~box] = np.nan               # keep only this cluster's cells
                name = f'rtv_{region.lower()}_{ym}_{abs(latc):.1f}{"N" if latc >= 0 else "S"}_{abs(float(lon[(x0 + x1) // 2])):.1f}{"W" if lon[(x0 + x1) // 2] < 0 else "E"}'
                meta = {'version': 1, 'name': name, 'lon0': float(lon[x0]), 'lat0': float(lat[y0]), 'dlon': dlon, 'dlat': dlat,
                        'dxKm': None, 'grid': 'HFRNet RTV', 'nx': int(x1 - x0 + 1), 'ny': int(y1 - y0 + 1), 'nFrames': nt,
                        't0': datetime.fromtimestamp(tstart, timezone.utc).isoformat().replace('+00:00', 'Z'),
                        'stepHours': 1.0, 'units': 'm/s', 'searchKm': 10.0, 'region': region, 'res': a.res,
                        'source': f'HFRNet RTV {region} {a.res} UWLS ({a.source}), NOAA NCEI'}
                path = os.path.join(a.npz_dir, name + '.npz')
                np.savez_compressed(path, u=U, v=V, lon=lon[x0:x1 + 1], lat=lat[y0:y1 + 1],
                                    time=tstart + 3600.0 * np.arange(nt), meta=json.dumps(meta))
                cov = np.isfinite(U).mean()
                print(f'  area {name}: {ext_x:.0f}×{ext_y:.0f} km, {meta["nx"]}×{meta["ny"]} cells, coverage {cov:.0%}')
                written.append(path)
    print(f'{len(written)} areas written to {a.npz_dir}')


if __name__ == '__main__':
    main()
