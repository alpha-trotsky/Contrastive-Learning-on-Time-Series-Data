"""Real data: the Jena "Weather" long-term-forecasting benchmark, as (u, v) windows.

Source: Max Planck Institute for Biogeochemistry roof station, Jena, 2020, one
row every 10 minutes (52,696 rows, 21 variables).  The CSV is downloaded on
first use into  <repo>/data/weather.csv  (kept out of git).

Each modality is a GROUP of variables, one representative per physical quantity
(Tpot, Tdew, VPmax, VPact, sh, H2OC, rho, Tlog are near-copies of T, humidity and
pressure and are left out, so that copies on both sides cannot make the task
trivial):

  temperature  T
  moisture     rh, VPdef
  pressure     p
  wind         wv, max. wv, sin(wd), cos(wd)       (wind direction is circular)
  radiation    SWDR, PAR
  rain         rain, raining                       (zero 96% of the time)

    u = past windows of the u_groups (past_len steps, every past_stride-th one),
        each column standardised with TRAINING statistics, concatenated
        column by column:  [col_1(t-P+1..t), col_2(...), ...]
    v = future window of the v_groups (future_len steps, every step), or with
        v_relative=True its change from the last observed value of the same
        column.  The last value is known at forecast time, so scores of a
        relative forecast equal those of (last value + forecast) against the
        absolute future: energy score and CRPS are unchanged by a shift.

Setup A (all groups -> future temperature) and setup B (cross-modal, e.g.
pressure + wind + moisture -> future rain) are just different group lists.

Time is split chronologically (default 70/10/20 train/val/test).  Inside the
training period, `bank_every`-th blocks of `bank_block` steps are held out of
training and form the BANK: the real futures a contrastive forecast resamples
from (and the climatology / residual-noise reference).  Every boundary between
periods (train / bank / val / test) is padded with a gap of past+future steps
that no window may touch, so no two windows from different periods share a
time step.  `bank: train` instead uses the training windows themselves.

    sample_pair(N)  random training windows (what experiment.run() trains on)
    split(name)     PairSampler over the 'val' / 'test' / 'bank' windows
"""
import math
import urllib.request
from pathlib import Path

import numpy as np
import torch

from modalities.base import PairSampler

URL = 'https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv'
DATA_PATH = Path(__file__).resolve().parents[2] / 'data' / 'weather.csv'

GROUPS = {
    'temperature': ['T (degC)'],
    'moisture': ['rh (%)', 'VPdef (mbar)'],
    'pressure': ['p (mbar)'],
    'wind': ['wv (m/s)', 'max. wv (m/s)', 'wd_sin', 'wd_cos'],
    'radiation': ['SWDR', 'PAR'],
    'rain': ['rain (mm)', 'raining (s)'],
}
# physical lower bound of a column (rain cannot be negative)
FLOORS = {'rain (mm)': 0.0, 'raining (s)': 0.0}

TRAIN, BANK, VAL, TEST, GAP = 0, 1, 2, 3, -1
SPLIT_IDS = {'train': TRAIN, 'bank': BANK, 'val': VAL, 'test': TEST}


def load_weather(path=DATA_PATH):
    """Cleaned weather table as {column: float64 array}, plus the timestamps."""
    import pandas as pd
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f'downloading {URL} -> {path}')
        urllib.request.urlretrieve(URL, path)
    df = pd.read_csv(path, encoding='latin-1')   # non-ASCII units in the header
    # the radiation headers carry mangled unit strings; keep the variable name only
    df.columns = [c.split(' (')[0] if c.startswith(('SWDR', 'PAR')) else c for c in df.columns]
    num = df.drop(columns='date').replace(-9999.0, np.nan)
    num = num.interpolate(limit_direction='both')                 # 81 sentinel cells
    rad = np.deg2rad(num['wd (deg)'].to_numpy())
    cols = {c: num[c].to_numpy(dtype=np.float64) for c in num.columns}
    cols['wd_sin'], cols['wd_cos'] = np.sin(rad), np.cos(rad)
    return cols, pd.to_datetime(df['date']).to_numpy()


class _WindowSampler(PairSampler):
    """(u, v) pairs at a fixed set of window start times."""

    def __init__(self, parent, starts):
        self.parent, self.starts = parent, torch.as_tensor(starts, dtype=torch.long)

    dim_u = property(lambda self: self.parent.dim_u)
    dim_v = property(lambda self: self.parent.dim_v)
    n_pairs = property(lambda self: len(self.starts))

    def draw_starts(self, N):
        """N window starts, without replacement when there are enough."""
        n = len(self.starts)
        idx = torch.randperm(n)[:N] if N <= n else torch.randint(0, n, (N,))
        return self.starts[idx]

    def sample_pair(self, N):
        return self.parent.pairs_at(self.draw_starts(N))

    def eval_set(self, N):
        """Fixed evaluation queries plus the no-learning reference forecasts.

        Returns u, v, and
          refs   {name: point forecast (N, 1, dim_v)}: persistence (last value
                 repeated) and seasonal-naive (same time one day earlier)
          proxy  {name: (N,)} difficulty proxies for conditional coverage
        """
        s = self.draw_starts(N)
        u, v = self.parent.pairs_at(s)
        return {'u': u, 'v': v, 'refs': self.parent.reference_forecasts(s),
                'proxy': self.parent.proxies(s), 'starts': s}


class WeatherModality(_WindowSampler):
    def __init__(self, u_groups, v_groups, past_len=144, future_len=36, past_stride=3,
                 fractions=(0.7, 0.1, 0.2), bank='heldout', bank_block=1008, bank_every=4,
                 v_relative=False, path=DATA_PATH):
        cols, self.time = load_weather(path)
        self.u_groups, self.v_groups = list(u_groups), list(v_groups)
        self.u_cols = [c for g in self.u_groups for c in GROUPS[g]]
        self.v_cols = [c for g in self.v_groups for c in GROUPS[g]]
        self.past_len, self.future_len, self.past_stride = past_len, future_len, past_stride
        self.v_relative = v_relative
        T = len(self.time)
        win = past_len + future_len

        # period label of every time step
        label = np.full(T, TRAIN)
        n_tr, n_va = int(fractions[0] * T), int(fractions[1] * T)
        label[n_tr:n_tr + n_va] = VAL
        label[n_tr + n_va:] = TEST
        if bank == 'heldout':
            for b, s in enumerate(range(0, n_tr, bank_block)):
                if b % bank_every == bank_every - 1:
                    label[s:min(s + bank_block, n_tr)] = BANK
        elif bank != 'train':
            raise ValueError(f"bank must be 'heldout' or 'train', not {bank!r}")
        self.bank_source = bank
        # gap of `win` steps after every change of period (either side is fine:
        # windows are win long, so a gap of win on one side keeps them apart)
        change = np.flatnonzero(np.diff(label) != 0) + 1
        self.label = label.copy()
        for c in change:
            self.label[c:c + win] = GAP

        # window starts per period: all win steps in one period
        lab = torch.as_tensor(self.label)
        w = lab.unfold(0, win, 1)                                     # (T-win+1, win)
        same = (w == w[:, :1]).all(1) & (w[:, 0] != GAP)
        self.split_starts = {name: torch.nonzero(same & (w[:, 0] == i)).flatten()
                             for name, i in SPLIT_IDS.items()}
        if bank == 'train':
            self.split_starts['bank'] = self.split_starts['train']
        train_mask = self.label == TRAIN

        # standardise every column with training-period statistics
        names = sorted(set(self.u_cols) | set(self.v_cols))
        self.mean = {c: cols[c][train_mask].mean() for c in names}
        self.std = {c: cols[c][train_mask].std() for c in names}
        z = lambda c: (cols[c] - self.mean[c]) / self.std[c]
        self.U = torch.as_tensor(np.stack([z(c) for c in self.u_cols], 1))   # (T, Cu)
        self.V = torch.as_tensor(np.stack([z(c) for c in self.v_cols], 1))   # (T, Cv)
        self.past_off = torch.arange(past_len - 1, -1, -past_stride).flip(0)  # ends at t
        self.future_off = torch.arange(past_len, win)
        self.v_scale = torch.tensor([self.std[c] for c in self.v_cols for _ in range(future_len)])
        floors = [(FLOORS[c] - self.mean[c]) / self.std[c] if c in FLOORS else -math.inf
                  for c in self.v_cols for _ in range(future_len)]
        has_floor = any(math.isfinite(f) for f in floors) and not v_relative
        self.v_floor = torch.tensor(floors) if has_floor else None
        super().__init__(self, self.split_starts['train'])

    @property
    def dim_u(self):
        return len(self.u_cols) * len(self.past_off)

    @property
    def dim_v(self):
        return len(self.v_cols) * self.future_len

    def pairs_at(self, starts):
        """u (N, dim_u), v (N, dim_v) for windows starting at `starts`."""
        s = torch.as_tensor(starts, dtype=torch.long)[:, None]
        u = self.U[s + self.past_off]                   # (N, P', Cu)
        v = self.V[s + self.future_off]                 # (N, F, Cv)
        if self.v_relative:
            v = v - self.V[s + self.past_len - 1]       # change from the last value
        return (u.transpose(1, 2).reshape(len(s), -1).double(),
                v.transpose(1, 2).reshape(len(s), -1).double())

    def reference_forecasts(self, starts):
        """Point forecasts that use no learning (standardised units)."""
        s = torch.as_tensor(starts, dtype=torch.long)[:, None]
        last = self.V[s[:, 0] + self.past_len - 1]                        # (N, Cv)
        persist = last[:, :, None].expand(-1, -1, self.future_len)
        day = 144                                                         # 1 day of 10-min steps
        refs = {'persistence': persist}
        if self.past_len >= day:
            refs['seasonal_naive'] = self.V[s + self.future_off - day].transpose(1, 2)
        if self.v_relative:
            refs = {k: r - last[:, :, None] for k, r in refs.items()}
        refs = {k: r.reshape(len(s), -1) for k, r in refs.items()}
        return {k: r.double()[:, None, :] for k, r in refs.items()}

    def proxies(self, starts):
        """Difficulty proxies from the past window (no oracle on real data):
        past_vol  std of the target variables' 10-min changes over the past day."""
        s = torch.as_tensor(starts, dtype=torch.long)[:, None]
        past = self.V[s + torch.arange(self.past_len)]                    # (N, P, Cv)
        return {'past_vol': past.diff(dim=1).std(dim=1).mean(1).double()}

    def split(self, name):
        return _WindowSampler(self, self.split_starts[name])

    def describe(self):
        """Sizes and date ranges of the periods (for the write-up)."""
        win = self.past_len + self.future_len
        out = {}
        for name, st in self.split_starts.items():
            if len(st):
                out[name] = {'windows': len(st),
                             'from': str(self.time[st[0]])[:16],
                             'to': str(self.time[st[-1] + win - 1])[:16]}
        return out
