#!/usr/bin/env python3
"""hfr_preprocess.py — HF radar surface currents (SQL) → gridded hourly frames for the WebGL viewer.

Reads either
  * a SQL dump file directly (MySQL/MariaDB `INSERT … VALUES` or PostgreSQL `COPY … FROM stdin`,
    optionally .gz) — streamed, so a 150 MB dump never needs a database server, or
  * a live database through an SQLAlchemy URL (--db).

Each hourly frame is gridded onto a regular lon/lat grid (linear interpolation within the
observation hull, masked wherever the nearest observation is farther than --max-gap-km, so
gaps are never filled by extrapolation).

Outputs for `--out data/hfr_202605`:
  data/hfr_202605.js    browser bundle: int16 u,v in mm/s, base64 (loads from file:// too)
  data/hfr_202605.npz   float32 u,v [m/s] with NaN gaps + coordinates, for hfr_spectra.py

Example:
  python tools/hfr_preprocess.py --sql 202605_current.sql --start 2026-05-01 --end 2026-05-08 \\
      --out data/hfr_202605
"""
import argparse
import base64
import gzip
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

MISSING = -32768          # int16 sentinel for "no data"
SCALE = 0.001             # m/s per int16 count (1 mm/s resolution, ±32.7 m/s range)

# ───────────────────────── time parsing ─────────────────────────

def parse_time(raw, offset_hours):
    """SQL time value → UTC epoch seconds. Accepts ISO strings or epoch seconds/milliseconds.
    Naive timestamps are taken as UTC + offset_hours (e.g. 7 for data stored in Thai local time)."""
    if raw is None:
        return None
    s = raw.strip().strip("'\"")
    try:
        x = float(s)
        return x / 1000.0 if x > 1e11 else x
    except ValueError:
        pass
    s = re.sub(r'\s+', ' ', s).replace('Z', '+00:00')      # e.g. '2026-04-30  17:00:00' (double space)
    if ' ' in s and 'T' not in s:
        s = s.replace(' ', 'T', 1)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc) - timedelta(hours=offset_hours)
    return dt.timestamp()


def to_float(raw):
    if raw is None:
        return math.nan
    s = raw.strip().strip("'\"")
    if s in ('', 'NULL', 'null', '\\N', 'NaN', 'nan'):
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan

# ───────────────────────── SQL dump streaming ─────────────────────────

IDENT = r'[`"]?(?:\w+[`"]?\.[`"]?)?(\w+)[`"]?'
RE_CREATE = re.compile(r'CREATE TABLE\s+(?:IF NOT EXISTS\s+)?' + IDENT + r'\s*\(', re.I)
RE_INSERT = re.compile(r'INSERT\s+INTO\s+' + IDENT + r'\s*(?:\(([^)]*)\))?\s*VALUES\s*(.*?);?\s*$', re.I | re.S)
RE_COPY = re.compile(r'COPY\s+' + IDENT + r'\s*\(([^)]*)\)\s+FROM\s+stdin', re.I)
RE_TUPLE = re.compile(r"\((?:[^()']|'(?:[^'\\]|\\.|'')*')*\)")
RE_FIELD = re.compile(r"\s*('(?:[^'\\]|\\.|'')*'|[^,]*?)\s*(?:,|$)")
NOT_COLUMN = ('PRIMARY', 'KEY', 'UNIQUE', 'CONSTRAINT', 'INDEX', 'FOREIGN', 'CHECK', 'FULLTEXT', 'SPATIAL')


def split_names(s):
    return [c.strip().strip('`"[] ') for c in s.split(',')]


def split_tuple(t):
    body = t[1:-1]
    out, pos = [], 0
    while pos <= len(body):
        m = RE_FIELD.match(body, pos)
        if not m or m.end() == pos:
            break
        v = m.group(1)
        out.append(None if v.upper() == 'NULL' else v)
        pos = m.end()
        if pos >= len(body):
            break
    return out


def iter_dump_rows(path, table):
    """Yield (table, {column: raw value}) from a MySQL/MariaDB or PostgreSQL dump."""
    opener = gzip.open if path.endswith('.gz') else open
    schemas = {}                       # table → column order from CREATE TABLE
    with opener(path, 'rt', encoding='utf-8', errors='replace') as fh:
        create, copy, stmt = None, None, None
        for line in fh:
            if copy is not None:                                  # PostgreSQL COPY block
                if line.startswith('\\.'):
                    copy = None
                    continue
                tname, cols = copy
                yield tname, dict(zip(cols, line.rstrip('\n').split('\t')))
                continue
            if create is not None:                                # inside CREATE TABLE
                tok = line.strip()
                if tok.startswith(')'):
                    create = None
                    continue
                name = tok.split()[0].strip('`"[],') if tok else ''
                if name and name.upper() not in NOT_COLUMN:
                    schemas[create].append(name)
                continue
            if stmt is None:
                m = RE_CREATE.match(line.lstrip())
                if m:
                    create = m.group(1)
                    schemas[create] = []
                    continue
                m = RE_COPY.match(line.lstrip())
                if m:
                    copy = (m.group(1), split_names(m.group(2)))
                    continue
                if not line.lstrip().upper().startswith('INSERT'):
                    continue
                stmt = line
            else:
                stmt += line
            if not stmt.rstrip().endswith(';'):
                continue                                          # statement continues
            m = RE_INSERT.match(stmt.strip())
            stmt = None
            if not m:
                continue
            tname = m.group(1)
            if table and tname != table:
                continue
            cols = split_names(m.group(2)) if m.group(2) else schemas.get(tname)
            if not cols:
                continue
            for t in RE_TUPLE.finditer(m.group(3)):
                yield tname, dict(zip(cols, split_tuple(t.group(0))))


def iter_db_rows(url, table, cols, t0, t1):
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        sys.exit('--db needs SQLAlchemy (pip install sqlalchemy plus a driver such as pymysql or psycopg2)')
    eng = create_engine(url)
    q = eng.dialect.identifier_preparer.quote
    sel = ', '.join(q(c) for c in cols)
    sql = text(f'SELECT {sel} FROM {q(table)} WHERE {q(cols[0])} >= :t0 AND {q(cols[0])} < :t1')
    with eng.connect() as conn:
        for row in conn.execution_options(stream_results=True).execute(sql, {'t0': t0, 't1': t1}):
            yield table, {c: (None if v is None else str(v)) for c, v in zip(cols, row)}

# ───────────────────────── gridding ─────────────────────────

def average_duplicates(lon, lat, u, v):
    """Average repeated observations at the same point within one frame (e.g. 30-min data)."""
    key = np.round(lon * 1e5).astype(np.int64) * 100000000 + np.round(lat * 1e5).astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv)
    mean = lambda a: np.bincount(inv, weights=a) / n
    return mean(lon), mean(lat), mean(u), mean(v)


def grid_frame(lon, lat, u, v, glon, glat, max_gap_km, method):
    lat0 = math.radians(float(np.mean(glat)))
    kx, ky = 111.32 * math.cos(lat0), 110.57              # km per degree (local equirectangular)
    pts = np.column_stack([lon * kx, lat * ky])
    GX, GY = np.meshgrid(glon * kx, glat * ky)
    targets = np.column_stack([GX.ravel(), GY.ravel()])
    dist, _ = cKDTree(pts).query(targets)
    out = []
    for comp in (u, v):
        g = griddata(pts, comp, targets, method=method) if len(pts) >= 4 else np.full(len(targets), np.nan)
        g[dist > max_gap_km] = np.nan                       # never extrapolate across gaps
        out.append(g.reshape(GX.shape))
    return out

# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--sql', help='SQL dump file (.sql or .sql.gz)')
    src.add_argument('--db', help='SQLAlchemy URL, e.g. mysql+pymysql://user:pw@host/db')
    ap.add_argument('--table', help='table name (default: any table with the required columns)')
    ap.add_argument('--where', action='append', default=[], metavar='COL=VALUE',
                    help="keep only rows where COL equals VALUE (repeatable), e.g. --where sitename=GULF")
    ap.add_argument('--cols', default='time,lat,lon,u_velocity,v_velocity',
                    help='column names for time, lat, lon, u, v (comma-separated)')
    ap.add_argument('--start', required=True, help='UTC start, e.g. 2026-05-01 or 2026-05-01T06:00')
    ap.add_argument('--end', required=True, help='UTC end (exclusive)')
    ap.add_argument('--step-hours', type=float, default=1.0, help='frame interval (default 1 h)')
    ap.add_argument('--time-offset-hours', type=float, default=0.0,
                    help='hours to subtract from naive timestamps to get UTC (7 if stored in Thai time)')
    ap.add_argument('--units', choices=['auto', 'm/s', 'cm/s'], default='auto')
    ap.add_argument('--bbox', type=float, nargs=4, metavar=('LON0', 'LON1', 'LAT0', 'LAT1'),
                    help='grid bounds (default: data extent)')
    ap.add_argument('--res', type=float, default=0.05, help='grid spacing in degrees (default 0.05° ≈ 5.5 km)')
    ap.add_argument('--max-gap-km', type=float, default=None, help='mask cells farther than this from data '
                    '(default 1.5 × grid spacing)')
    ap.add_argument('--method', choices=['linear', 'nearest', 'cubic'], default='linear')
    ap.add_argument('--native-xy', metavar='XCOL,YCOL',
                    help='place vectors on their native Cartesian grid (e.g. xdistance,ydistance in km) '
                         'instead of interpolating: exact values, no smoothing — use for spectra')
    ap.add_argument('--native-dx', type=float, default=None, help='native grid spacing in the XCOL/YCOL '
                    'units (default: smallest spacing found in the data)')
    ap.add_argument('--out', required=True, help='output path prefix, e.g. data/hfr_202605')
    ap.add_argument('--npz', help='path of the .npz for spectral analysis (default: <out>.npz)')
    a = ap.parse_args()

    tcol, latcol, loncol, ucol, vcol = [c.strip() for c in a.cols.split(',')]
    where = [w.split('=', 1) for w in a.where]
    if any(len(w) != 2 for w in where):
        sys.exit('--where expects COL=VALUE')
    t0 = parse_time(a.start, 0.0)
    t1 = parse_time(a.end, 0.0)
    if t0 is None or t1 is None or t1 <= t0:
        sys.exit('Invalid --start/--end')
    step = a.step_hours * 3600.0
    nt = int(math.ceil((t1 - t0) / step))

    # 1) Stream rows inside the window.
    off = a.time_offset_hours * 3600.0               # DB stores naive times in UTC + offset
    db_time = lambda t: datetime.fromtimestamp(t + off, timezone.utc).replace(tzinfo=None)
    rows = iter_db_rows(a.db, a.table, [tcol, latcol, loncol, ucol, vcol],
                        db_time(t0 - step / 2), db_time(t1)) if a.db \
        else iter_dump_rows(a.sql, a.table)
    xcol, ycol = (a.native_xy.split(',') + [None])[:2] if a.native_xy else (None, None)
    T, LAT, LON, U, V, XS, YS = [], [], [], [], [], [], []
    seen = kept = 0
    tables = set()
    for tname, r in rows:
        if tcol not in r or ucol not in r:
            continue
        seen += 1
        if any((r.get(c) or '').strip("'\" ") != v for c, v in where):
            continue
        t = parse_time(r.get(tcol), a.time_offset_hours)
        if t is None or t < t0 - step / 2 or t >= t1 - step / 2:
            continue
        vals = [to_float(r.get(c)) for c in (latcol, loncol, ucol, vcol)]
        if xcol:
            vals += [to_float(r.get(xcol)), to_float(r.get(ycol))]
        if any(math.isnan(x) for x in vals):
            continue
        T.append(t); LAT.append(vals[0]); LON.append(vals[1]); U.append(vals[2]); V.append(vals[3])
        if xcol:
            XS.append(vals[4]); YS.append(vals[5])
        kept += 1
        tables.add(tname)
        if seen % 500000 == 0:
            print(f'  … {seen:,} rows read, {kept:,} in window', file=sys.stderr)
    if not kept:
        sys.exit(f'No rows in the window ({seen:,} rows read). Check --cols, --table, --start/--end and --time-offset-hours.')
    T, LAT, LON, U, V = map(np.asarray, (T, LAT, LON, U, V))
    print(f'{seen:,} rows read from {", ".join(sorted(tables))}; {kept:,} in the window')

    # 2) Units.
    units = a.units
    if units == 'auto':
        units = 'cm/s' if np.nanmedian(np.hypot(U, V)) > 3.0 else 'm/s'   # ocean currents are rarely > 3 m/s
        print(f'units: auto-detected {units} (median speed {np.nanmedian(np.hypot(U, V)):.2f})')
    if units == 'cm/s':
        U, V = U / 100.0, V / 100.0

    # 3) Grid.
    frame = np.rint((T - t0) / step).astype(int)
    if xcol:
        return export(a, *native_grid(a, T, LAT, LON, U, V, np.asarray(XS), np.asarray(YS), frame, nt), t0, step, nt)
    if a.bbox:
        lon0, lon1, lat0, lat1 = a.bbox
    else:
        lon0, lon1, lat0, lat1 = LON.min(), LON.max(), LAT.min(), LAT.max()
    glon = np.arange(lon0, lon1 + a.res / 2, a.res)
    glat = np.arange(lat0, lat1 + a.res / 2, a.res)          # row 0 = south
    max_gap = a.max_gap_km or 1.5 * a.res * 111.0
    u_out = np.full((nt, len(glat), len(glon)), np.nan, np.float32)
    v_out = np.full_like(u_out, np.nan)
    for k in range(nt):
        sel = frame == k
        if sel.sum() < 4:
            continue
        lo, la, uu, vv = average_duplicates(LON[sel], LAT[sel], U[sel], V[sel])
        u_out[k], v_out[k] = grid_frame(lo, la, uu, vv, glon, glat, max_gap, a.method)
    cover = np.isfinite(u_out).mean(axis=(1, 2))
    print(f'grid {len(glon)}×{len(glat)} at {a.res}°, {nt} frames, coverage min/mean/max '
          f'{cover.min():.0%}/{cover.mean():.0%}/{cover.max():.0%}')
    export(a, u_out, v_out, glon, glat, a.res, a.res, None, t0, step, nt)


def native_grid(a, T, LAT, LON, U, V, X, Y, frame, nt):
    """Put each vector on its native Cartesian node; lon/lat of the nodes from a linear fit."""
    dx = a.native_dx or float(np.min(np.diff(np.unique(np.round(X, 3)))))
    ix = np.rint((X - X.min()) / dx).astype(int)
    iy = np.rint((Y - Y.min()) / dx).astype(int)
    nx, ny = ix.max() + 1, iy.max() + 1
    A = np.column_stack([np.ones_like(ix), ix, iy]).astype(float)
    (lo0, lo_x, lo_y), *_ = np.linalg.lstsq(A, LON, rcond=None)
    (la0, la_x, la_y), *_ = np.linalg.lstsq(A, LAT, rcond=None)
    rot = max(abs(lo_y / lo_x), abs(la_x / la_y))
    res = max(np.abs(A @ [lo0, lo_x, lo_y] - LON).max() / lo_x, np.abs(A @ [la0, la_x, la_y] - LAT).max() / la_y)
    print(f'native grid {nx}×{ny}, spacing {dx:g} (xy units) = {lo_x:.5f}° lon × {la_y:.5f}° lat; '
          f'axis skew {rot:.1e}, max node misfit {res:.2f} cells')
    if rot > 0.02:
        print('  warning: native grid is rotated relative to lon/lat; the overlay will be slightly displaced')
    u_out = np.full((nt, ny, nx), np.nan, np.float32)
    v_out = np.full_like(u_out, np.nan)
    flat = (frame * ny + iy) * nx + ix                       # average duplicates per node and frame
    for comp, out in ((U, u_out), (V, v_out)):
        s = np.bincount(flat, weights=comp, minlength=nt * ny * nx)
        n = np.bincount(flat, minlength=nt * ny * nx)
        out.ravel()[n > 0] = (s[n > 0] / n[n > 0]).astype(np.float32)
    cover = np.isfinite(u_out).mean(axis=(1, 2))
    print(f'{nt} frames, node coverage min/mean/max {cover.min():.0%}/{cover.mean():.0%}/{cover.max():.0%}')
    glon = lo0 + lo_x * np.arange(nx)
    glat = la0 + la_y * np.arange(ny)
    return u_out, v_out, glon, glat, float(lo_x), float(la_y), float(dx)


def export(a, u_out, v_out, glon, glat, dlon, dlat, dx_km, t0, step, nt):
    meta = {
        'version': 1, 'name': a.out.split('/')[-1],
        'lon0': float(glon[0]), 'lat0': float(glat[0]), 'dlon': dlon, 'dlat': dlat,
        'dxKm': dx_km, 'grid': 'native' if dx_km else 'interpolated',
        'nx': len(glon), 'ny': len(glat), 'nFrames': nt,
        't0': datetime.fromtimestamp(t0, timezone.utc).isoformat().replace('+00:00', 'Z'),
        'stepHours': a.step_hours, 'scale': SCALE, 'missing': MISSING, 'units': 'm/s',
        'layout': 'int16[frame][row (south→north)][col (west→east)][u,v]',
        'source': (a.sql.replace('\\', '/').split('/')[-1] if a.sql else a.table), 'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    q = np.stack([u_out, v_out], axis=-1)
    packed = np.where(np.isfinite(q), np.clip(np.rint(q / SCALE), -32767, 32767), MISSING).astype('<i2')
    b64 = base64.b64encode(packed.tobytes()).decode('ascii')
    with open(a.out + '.js', 'w') as fh:
        fh.write('/* HF radar surface currents — generated by tools/hfr_preprocess.py */\n')
        fh.write('(window.HFR_DATA = window.HFR_DATA || []).push({meta: ')
        fh.write(json.dumps(meta))
        fh.write(', data: "' + b64 + '"});\n')
    npz = a.npz or a.out + '.npz'
    np.savez_compressed(npz, u=u_out, v=v_out, lon=glon, lat=glat,
                        time=t0 + step * np.arange(nt), meta=json.dumps(meta))
    print(f'wrote {a.out}.js ({len(b64) / 1e6:.1f} MB) and {npz}')


if __name__ == '__main__':
    main()
