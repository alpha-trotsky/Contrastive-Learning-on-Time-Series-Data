"""Real heavy-tailed series as (past u, future v) window pairs with
chronological train / bank / test splits.

    from modalities.heavy_real import load, DEFAULTS
    d = load('btc', **DEFAULTS['btc'])
    d['u_train'], d['v_train'], d['u_bank'], d['v_bank'], d['u_test'], d['v_test']

DEFAULTS = {
    'btc':  dict(past_len=64, future_len=16, stride=8),   # 1-minute steps
    'rain': dict(past_len=64, future_len=16, stride=4),   # 10-minute steps
    'flow': dict(past_len=64, future_len=16, stride=4),   # 15-minute steps
}

Datasets (details and diagnostics: docs/real_heavy_tail_data.md)

  btc   Binance spot BTCUSDT 1-minute klines, 2019-01 .. 2024-12 (72 monthly
        zips from data.binance.vision, sha256-checked).  Target = 1-minute log
        return of the close.  A return exists only between two consecutive
        minutes; a missing minute (exchange outage) starts a new segment.
        Extra u channel (extra_channels=True): log(1 + traded volume in BTC).
  rain  Max Planck Institute for Biogeochemistry roof station, Jena, 10-minute
        data 2003-11 .. 2025-12 (bgc-jena.mpg.de, CC-BY-4.0; the 2020 slice is
        the "Weather" benchmark already used in modalities/weather.py).
        Target = rain (mm per 10 minutes), non-negative, zero ~96% of the time.
        Extra u channels: relative humidity rh (%), air pressure p (mbar).
  flow  USGS 02336300 Peachtree Creek at Atlanta, GA (a flashy urban stream),
        instantaneous discharge (parameter 00060, ft3/s) on its 15-minute grid,
        1986-10 .. 2025-12 (waterservices.usgs.gov).  Target = discharge.
        No extra channels.

Conventions

  * A window is past_len + future_len CONSECUTIVE steps of one segment: no
    window spans a missing time step.  Window starts lie on the grid
    0, stride, 2*stride, ... of the cleaned series (default stride = future_len,
    i.e. non-overlapping futures).
  * Splits are chronological in time steps of the cleaned series: train = first
    fractions[0], bank = next fractions[1], test = the rest.  The first
    past_len + future_len steps after each boundary belong to no split, so two
    windows of different splits never share a time step.
  * v (n, future_len) = future of the target divided by ONE constant v_scale =
    root mean square of the target over the training period (all its time
    steps, zeros included; no centring).  Nothing else is done to v: no
    clipping, winsorising, rank transform or per-window normalisation.
  * u (n, past_len * n_u_channels) = past of the u channels, each standardised
    with its training-period mean and std (u_mean, u_std), laid out channel by
    channel: [target(t-P..t-1), extra_1(t-P..t-1), ...].  The first past_len
    columns are always the past of the target, so the past in v's units is
        u[:, :past_len] * u_std[0] / v_scale + u_mean[0] / v_scale.
    u is not clipped either.
  * max_windows caps EACH split: a split with more windows keeps max_windows of
    them, evenly spaced in time.

The cleaned series is cached in <repo>/data/<name>.npz; raw files are
downloaded into <repo>/data/raw/ only when that cache is missing.  data/ is
git-ignored.  Downloaded files are only parsed as CSV / tab-separated text.
"""
import hashlib
import io
import time as _time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
RAW_DIR = DATA_DIR / 'raw'

DEFAULTS = {
    'btc': dict(past_len=64, future_len=16, stride=8),
    'rain': dict(past_len=64, future_len=16, stride=4),
    'flow': dict(past_len=64, future_len=16, stride=4),
}

BINANCE_URL = 'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/'
BINANCE_MONTHS = [f'{y}-{m:02d}' for y in range(2019, 2025) for m in range(1, 13)]
JENA_URL = 'https://www.bgc-jena.mpg.de/wetter/data/'
JENA_PARTS = (['2003b'] + [f'{y}{h}' for y in range(2004, 2024) for h in 'ab'] + ['2024', '2025'])
USGS_URL = ('https://waterservices.usgs.gov/nwis/iv/?format=rdb&sites={site}'
            '&parameterCd=00060&startDT={y}-01-01&endDT={y}-12-31')
USGS_SITE = '02336300'
USGS_YEARS = range(1986, 2026)

DESCRIPTIONS = {
    'btc': 'Binance spot BTCUSDT, 1-minute log returns of the close, 2019-01..2024-12',
    'rain': 'MPI-BGC Jena roof station, rain (mm per 10 min), 2003-11..2025-12',
    'flow': 'USGS 02336300 Peachtree Creek at Atlanta GA, 15-minute discharge (ft3/s), 1986-10..2025-12',
}


# ---------------------------------------------------------------------------
# download helpers
# ---------------------------------------------------------------------------

def _fetch(url, path, check=None, tries=4):
    """Download url -> path (atomic) unless path exists; check(bytes) validates."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                blob = r.read()
            if check is not None:
                check(blob)
            tmp = path.with_name(path.name + '.part')
            tmp.write_bytes(blob)
            tmp.replace(path)
            return path
        except Exception as e:                      # retry transient failures
            if attempt == tries - 1:
                raise RuntimeError(f'could not download {url}: {e}') from e
            _time.sleep(2 + 3 * attempt)


def _check_zip(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if z.testzip() is not None:
            raise ValueError('corrupt zip')


def _check_rdb(blob):
    if not blob.startswith(b'#'):
        raise ValueError('not a USGS rdb file: ' + blob[:80].decode('latin-1'))


def _single_csv(zip_path):
    """The one CSV inside a zip, as bytes (nothing is extracted to disk)."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith('.csv')]
        if len(names) != 1:
            raise ValueError(f'{zip_path}: expected one csv, found {z.namelist()}')
        return z.read(names[0])


def _segments(time_s, step_s):
    """Segment id per row: a new segment wherever the time step is not step_s."""
    brk = np.diff(time_s) != step_s
    return np.concatenate([[0], np.cumsum(brk)]).astype(np.int32)


# ---------------------------------------------------------------------------
# builders: raw files -> cleaned series
#   x (T, C) float64, column 0 = target;  time (T,) int64 epoch seconds;
#   seg (T,) int32, constant on runs of consecutive time steps
# ---------------------------------------------------------------------------

def _build_btc():
    import pandas as pd
    raw = RAW_DIR / 'binance'
    ts, close, vol = [], [], []
    for month in BINANCE_MONTHS:
        name = f'BTCUSDT-1m-{month}.zip'
        zp = _fetch(BINANCE_URL + name, raw / name, _check_zip)
        cs = _fetch(BINANCE_URL + name + '.CHECKSUM', raw / (name + '.CHECKSUM'))
        if hashlib.sha256(zp.read_bytes()).hexdigest() != cs.read_text().split()[0]:
            raise ValueError(f'{zp}: sha256 does not match the published checksum')
        df = pd.read_csv(io.BytesIO(_single_csv(zp)), header=None, usecols=[0, 4, 5])
        df = df[pd.to_numeric(df[0], errors='coerce').notna()]        # a header row, if any
        t = df[0].to_numpy(np.int64)
        t = np.where(t > 10 ** 14, t // 1000, t)                      # microseconds (2025+) -> ms
        ts.append(t)
        close.append(df[4].to_numpy(np.float64))
        vol.append(df[5].to_numpy(np.float64))
    ts, close, vol = np.concatenate(ts), np.concatenate(close), np.concatenate(vol)
    order = np.argsort(ts, kind='stable')
    ts, close, vol = ts[order], close[order], vol[order]
    keep = np.concatenate([[True], np.diff(ts) > 0]) & np.isfinite(close) & (close > 0)
    ts, close, vol = ts[keep] // 1000, close[keep], vol[keep]
    # return of bar i = log(close_i / close_{i-1}), defined when bar i-1 is the previous minute
    ok = np.diff(ts) == 60
    ret = np.diff(np.log(close))[ok]
    t_ret, v_ret = ts[1:][ok], vol[1:][ok]
    x = np.stack([ret, np.log1p(v_ret)], 1)
    return x, t_ret, _segments(t_ret, 60), ['log_return', 'log1p_volume']


def _build_rain():
    import pandas as pd
    raw = RAW_DIR / 'jena'
    cols = ['rain (mm)', 'rh (%)', 'p (mbar)']
    frames = []
    for part in JENA_PARTS:
        name = f'mpi_roof_{part}.zip'
        zp = _fetch(JENA_URL + name, raw / name, _check_zip)
        df = pd.read_csv(io.BytesIO(_single_csv(zp)), encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]
        t = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
        out = df[cols].apply(pd.to_numeric, errors='coerce')
        out.insert(0, 't', t)
        frames.append(out)
    a = pd.concat(frames, ignore_index=True).dropna(subset=['t'])
    sec = ((a['t'] - pd.Timestamp('1970-01-01')) // pd.Timedelta(seconds=1)).to_numpy(np.int64)
    a = a.assign(sec=sec)
    a = a[a['sec'] % 600 == 0]                                        # off-grid logger rows
    a = a.drop_duplicates('sec').sort_values('sec')
    x = a[cols].to_numpy(np.float64)
    # -9999 is the logger's missing-value sentinel; a row with any missing or
    # impossible value is dropped (it then separates two segments)
    good = (np.isfinite(x).all(1) & (x[:, 0] >= 0) & (x[:, 0] < 100)
            & (x[:, 1] >= 0) & (x[:, 1] <= 110) & (x[:, 2] > 800) & (x[:, 2] < 1100))
    x, sec = x[good], a['sec'].to_numpy()[good]
    return x, sec, _segments(sec, 600), ['rain_mm', 'rh_pct', 'p_mbar']


def _build_flow():
    import pandas as pd
    raw = RAW_DIR / 'usgs'
    frames = []
    for y in USGS_YEARS:
        fp = _fetch(USGS_URL.format(site=USGS_SITE, y=y),
                    raw / f'iv_{USGS_SITE}_00060_{y}.rdb', _check_rdb)
        df = pd.read_csv(fp, sep='\t', comment='#', dtype=str)
        df = df.iloc[1:]                                              # rdb column-format row
        qcol = [c for c in df.columns if c.endswith('_00060')]
        if len(df) == 0 or len(qcol) != 1:
            continue
        frames.append(pd.DataFrame({'t': df['datetime'], 'tz': df['tz_cd'],
                                    'q': pd.to_numeric(df[qcol[0]], errors='coerce')}))
    a = pd.concat(frames, ignore_index=True)
    # local clock time -> UTC seconds with the fixed offsets the file states
    off = a['tz'].map({'EST': 5, 'EDT': 4})
    t = pd.to_datetime(a['t'], format='%Y-%m-%d %H:%M', errors='coerce')
    ok = (off.notna() & t.notna() & a['q'].notna()).to_numpy()
    a, t, off = a[ok], t[ok], off[ok]
    sec = ((t - pd.Timestamp('1970-01-01')) // pd.Timedelta(seconds=1)).to_numpy(np.int64)
    sec = sec + off.to_numpy(np.int64) * 3600
    q = a['q'].to_numpy(np.float64)
    keep = (sec % 900 == 0) & (q > 0)                                 # 15-minute grid, positive flow
    sec, q = sec[keep], q[keep]
    order = np.argsort(sec, kind='stable')
    sec, q = sec[order], q[order]
    first = np.concatenate([[True], np.diff(sec) > 0])
    sec, q = sec[first], q[first]
    return q[:, None], sec, _segments(sec, 900), ['discharge_cfs']


_BUILDERS = {'btc': _build_btc, 'rain': _build_rain, 'flow': _build_flow}


def load_series(name, rebuild=False):
    """Cleaned series {x (T, C), time (T,), seg (T,), channels}, cached in data/<name>.npz."""
    if name not in _BUILDERS:
        raise ValueError(f'name must be one of {sorted(_BUILDERS)}, not {name!r}')
    cache = DATA_DIR / f'{name}.npz'
    if cache.exists() and not rebuild:
        z = np.load(cache)
        return {'x': z['x'], 'time': z['time'], 'seg': z['seg'], 'channels': [str(c) for c in z['channels']]}
    x, t, seg, channels = _BUILDERS[name]()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.stem + '.tmp.npz')
    np.savez(tmp, x=x, time=t, seg=seg, channels=np.array(channels))
    tmp.replace(cache)
    return {'x': x, 'time': t, 'seg': seg, 'channels': channels}


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------

def load(name, past_len, future_len, stride=None, fractions=(0.5, 0.3, 0.2), max_windows=None,
         extra_channels=True):
    """(past u, future v) windows of a real heavy-tailed series.

    name            'btc' | 'rain' | 'flow'            (see DEFAULTS and the module docstring)
    stride          step between consecutive window starts (default future_len)
    fractions       chronological (train, bank, test) shares of the time steps
    max_windows     cap per split (evenly spaced in time), None = keep all
    extra_channels  False: u is the past of the target only

    Returns a dict with float64 torch tensors
        u_train, v_train, u_bank, v_bank, u_test, v_test    (n, past_len*n_u_channels), (n, future_len)
    int64 tensors t_train, t_bank, t_test (epoch seconds, UTC, of the first future step) and
    metadata: u_mean, u_std (per u channel), v_scale, u_channels, n_u_channels,
    name, description, n_steps, n_segments, n_train, n_bank, n_test, range_<split>.
    """
    s = load_series(name)
    x, t, seg = s['x'], s['time'], s['seg']
    if not extra_channels:
        x = x[:, :1]
    stride = future_len if stride is None else int(stride)
    P, F = int(past_len), int(future_len)
    win, T = P + F, len(x)
    if abs(sum(fractions) - 1) > 1e-9 or len(fractions) != 3:
        raise ValueError('fractions must be (train, bank, test) summing to 1')
    b1 = int(fractions[0] * T)
    b2 = int((fractions[0] + fractions[1]) * T)

    starts = np.arange(0, T - win + 1, stride)
    starts = starts[seg[starts] == seg[starts + win - 1]]             # inside one segment
    split_starts = {
        'train': starts[starts + win <= b1],
        'bank': starts[(starts >= b1 + win) & (starts + win <= b2)],
        'test': starts[starts >= b2 + win],
    }
    if max_windows is not None:
        for k, st in split_starts.items():
            if len(st) > max_windows:
                split_starts[k] = st[np.linspace(0, len(st) - 1, int(max_windows)).round().astype(np.int64)]

    u_mean, u_std = x[:b1].mean(0), x[:b1].std(0)
    v_scale = float(np.sqrt(np.mean(x[:b1, 0] ** 2)))
    xt = torch.as_tensor(x)                                           # (T, C) float64, shared memory
    u_mean_t, u_std_t = torch.as_tensor(u_mean), torch.as_tensor(u_std)
    past_off, fut_off = torch.arange(P), torch.arange(P, win)

    out = {}
    for k, st in split_starts.items():
        st_t = torch.as_tensor(st, dtype=torch.long)[:, None]
        u = (xt[st_t + past_off] - u_mean_t) / u_std_t                # (n, P, C)
        out['u_' + k] = u.transpose(1, 2).reshape(len(st), -1)
        out['v_' + k] = xt[:, 0][st_t + fut_off] / v_scale            # (n, F)
        out['t_' + k] = torch.as_tensor(t[st + P]) if len(st) else torch.zeros(0, dtype=torch.long)
        out['n_' + k] = len(st)
        out['range_' + k] = tuple(str(np.datetime64(int(v), 's')) for v in
                                  (t[st[0]], t[st[-1] + win - 1])) if len(st) else None
    out.update(
        u_mean=u_mean_t, u_std=u_std_t, v_scale=v_scale,
        u_channels=list(s['channels'][:x.shape[1]]), n_u_channels=x.shape[1],
        name=name, description=DESCRIPTIONS[name],
        past_len=P, future_len=F, stride=stride, fractions=tuple(fractions),
        n_steps=T, n_segments=int(seg[-1]) + 1,
    )
    return out
