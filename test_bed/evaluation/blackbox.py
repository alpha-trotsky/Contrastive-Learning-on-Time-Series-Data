"""Encoder-agnostic forecast evaluation for experiment.py.

Works for any model with encode_u / encode_v (linear or MLP): the contrastive
model's forecast is a resampling of real futures with weights ∝ exp(score)
(see evaluation/predictors.py), scored against the realised future with the
proper scoring rules in evaluation/scoring.py.

Reference forecasters:
  climatology  random futures, ignores the past (the no-skill floor)
  oracle       the true p(v|u), when the modality can provide it:
               `oracle_samples(u, M)` (non-Gaussian domains) or the Gaussian
               conditional from C_uu / C_vv / C_uv (the GP test bed)

The evaluation data are drawn under a fixed seed inside torch.random.fork_rng,
so calling this never changes the training run's random stream.

Real data: test queries and bank must not come from the training period.  A
modality that has held-out periods exposes  split(name) -> PairSampler  for
name in {'val', 'test', 'bank'}; experiment.run() then evaluates on 'val'
during training and on 'test' at the end, resampling futures from 'bank'.
Without split(), everything is drawn from the modality itself (synthetic data,
where every draw is fresh).  A test sampler with  eval_set(n)  (real data,
modalities/weather.py) also supplies no-learning reference forecasts
(persistence, seasonal-naive), scored alongside climatology, and difficulty
proxies for conditional coverage (see `detail`).

Regression baselines (models/mlp_regressor.py, any depth) are scored by
sampling their own predictive, with residual noise estimated on the bank.
"""
import torch

from evaluation.predictors import (sir_samples, clip_score_fn, climatology_samples, gaussian_samples,
                                   regressor_samples)
from evaluation.scoring import score_forecast, add_skills, crps, pit


def _sample(modality, n, chunk=10000):
    if hasattr(modality, 'n_pairs'):      # a fixed set of windows (real data): no repeats
        return tuple(x.double() for x in modality.sample_pair(min(n, modality.n_pairs)))
    us, vs = zip(*(modality.sample_pair(min(chunk, n - i)) for i in range(0, n, chunk)))
    return torch.cat(us).double(), torch.cat(vs).double()


def oracle_samples(modality, u, M):
    """Exact draws from p(v|u) if the modality can provide them, else None."""
    if hasattr(modality, 'oracle_samples'):
        return modality.oracle_samples(u, M)
    try:
        C_uu, C_vv, C_uv = (c().double() for c in (modality.C_uu, modality.C_vv, modality.C_uv))
    except (AttributeError, NotImplementedError):
        return None
    K = torch.linalg.solve(C_uu, C_uv)
    return gaussian_samples(u @ K, C_vv - C_uv.T @ K, M)


class BlackBoxEval:
    """Fixed test queries + bank, scored repeatedly during / after training.

    loss_cfg   : the experiment's loss config (picks the score: cosine,
                 normalised cosine or L2; 'mse' uses the regression head
                 W_u^T W_v plus Gaussian residual noise, linear models only)
    n_test     : number of test queries
    bank       : number of futures the forecast resamples from (fresh pairs,
                 not training data)
    M          : forecast samples per query
    test_sampler / bank_sampler : PairSamplers for the test queries / bank
                 (default: the modality itself)
    """

    def __init__(self, modality, loss_cfg, n_test=500, bank=20000, M=300, seed=4321,
                 test_sampler=None, bank_sampler=None):
        self.loss_cfg, self.M = loss_cfg, M
        test_sampler = test_sampler or modality
        bank_sampler = bank_sampler or modality
        self.refs, self.proxy = {}, {}
        self.v_floor = getattr(modality, 'v_floor', None)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            if hasattr(test_sampler, 'eval_set'):
                e = test_sampler.eval_set(n_test)
                self.u, self.y = e['u'].double(), e['v'].double()
                self.refs, self.proxy = e['refs'], e['proxy']
            else:
                self.u, self.y = _sample(test_sampler, n_test)
            self.u_bank, self.v_bank = _sample(bank_sampler, bank)
            self.oracle = oracle_samples(test_sampler, self.u, M)
            self.clim = climatology_samples(self.v_bank, self.u.shape[0], M)
        self.refs = {'climatology': self.clim, **self.refs}
        if self.oracle is not None:
            self.refs['oracle'] = self.oracle
        self.ref = {k: score_forecast(r, self.y) for k, r in self.refs.items()}

    @torch.no_grad()
    def model_samples(self, model):
        """Forecast samples (N, M, d) and, for contrastive models, the ESS."""
        with torch.random.fork_rng():
            torch.manual_seed(0)
            if hasattr(model, 'predict'):           # regression baselines
                return regressor_samples(model, self.u, self.u_bank, self.v_bank, self.M), None
            if self.loss_cfg['type'] == 'mse':
                W = (model.u_encoder.weight.T @ model.v_encoder.weight).double()
                r = self.v_bank - self.u_bank @ W
                return gaussian_samples(self.u @ W, r.T @ r / r.shape[0], self.M), None
            score = clip_score_fn(model, self.loss_cfg.get('inner_product', True),
                                  self.loss_cfg.get('normalize', False))
            # resample the bank itself (float64), not a float32 copy: exact values
            # matter for point masses such as dry futures (rain = 0)
            return sir_samples(score, self.u, self.v_bank, self.M)

    def _eval_samples(self, model):
        was_training = model.training
        model.eval()
        samples, ess = self.model_samples(model)
        model.train(was_training)
        return samples, ess

    def __call__(self, model):
        """Flat dict of metrics for logging (prefix 'bb/')."""
        samples, ess = self._eval_samples(model)
        table = add_skills({'model': score_forecast(samples.double(), self.y), **self.ref})
        m = table['model']
        out = {f'bb/{k}': m[k] for k in ('energy', 'energy_skill', 'crps', 'mse_mean',
                                         'cov50', 'cov90', 'width90', 'cal_err')}
        if 'oracle' in table:
            o = table['oracle']
            out['bb/energy_vs_oracle'] = m['energy'] / o['energy'] - 1
            out['bb/width90_vs_oracle'] = m['width90'] / o['width90']
        if ess is not None:
            out['bb/ess_median'] = ess.median().item()
            out['bb/ess_min'] = ess.min().item()
        for k, r in table.items():
            if k not in ('model', 'climatology', 'oracle'):
                out[f'bb/ref_{k}_energy_skill'] = r['energy_skill']
        return out

    def detail(self, model):
        """Per-forecaster detail for the final report (lists, JSON-ready):
          scores        all scalar scores of score_forecast, with skills
          crps_step     CRPS per output coordinate (horizon step), averaged over queries
          bias_step     mean error (forecast mean - truth) per output coordinate
          cov90_query   per-query fraction of coordinates inside the 90% band
                        (bin by `proxy` for conditional coverage)
          cov90_step    90% band coverage per output coordinate (horizon step)
          pit_hist      10-bin histogram of PIT values (flat = calibrated)
          below_floor   fraction of forecast samples below the physical floor
                        (e.g. negative rain), when the modality has one
          brier_any     Brier score of the event "any coordinate above the floor"
                        (e.g. any rain in the next hours), when it has a floor
        Also the ESS per query, and the proxies themselves."""
        samples, ess = self._eval_samples(model)
        S = {'model': samples.double(), **self.refs}
        table = add_skills({k: score_forecast(s, self.y) for k, s in S.items()})
        out = {'forecasters': {}, 'proxy': {k: p.tolist() for k, p in self.proxy.items()}}
        q = torch.tensor([0.05, 0.95], dtype=torch.float64)
        for k, s in S.items():
            d = {'scores': table[k],
                 'crps_step': crps(s, self.y).mean(0).tolist(),
                 'bias_step': (s.mean(1) - self.y).mean(0).tolist()}
            if s.shape[1] > 1:
                lo, hi = torch.quantile(s, q, dim=1)
                inside = ((self.y >= lo) & (self.y <= hi)).double()
                d['cov90_query'] = inside.mean(1).tolist()
                d['cov90_step'] = inside.mean(0).tolist()
                d['pit_hist'] = torch.histc(pit(s, self.y).float(), bins=10, min=0, max=1).tolist()
            if self.v_floor is not None:
                fl = self.v_floor.to(s)
                d['below_floor'] = (s < fl - 1e-6).double().mean().item()
                p_event = (s > fl + 1e-6).any(-1).double().mean(1)
                event = (self.y > fl + 1e-6).any(-1).double()
                d['brier_any'] = ((p_event - event) ** 2).mean().item()
                d['event_rate'] = event.mean().item()
            out['forecasters'][k] = d
        if ess is not None:
            out['ess'] = ess.tolist()
        return out
