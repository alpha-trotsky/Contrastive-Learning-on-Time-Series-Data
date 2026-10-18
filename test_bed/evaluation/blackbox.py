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
where every draw is fresh).
"""
import torch

from evaluation.predictors import sir_samples, clip_score_fn, climatology_samples, gaussian_samples
from evaluation.scoring import score_forecast, add_skills


def _sample(modality, n, chunk=10000):
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
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.u, self.y = _sample(test_sampler, n_test)
            self.u_bank, self.v_bank = _sample(bank_sampler, bank)
            self.oracle = oracle_samples(test_sampler, self.u, M)
            self.clim = climatology_samples(self.v_bank, n_test, M)
        ref = {'climatology': score_forecast(self.clim, self.y)}
        if self.oracle is not None:
            ref['oracle'] = score_forecast(self.oracle, self.y)
        self.ref = ref

    @torch.no_grad()
    def model_samples(self, model):
        """Forecast samples (N, M, d) and, for contrastive models, the ESS."""
        with torch.random.fork_rng():
            torch.manual_seed(0)
            if self.loss_cfg['type'] == 'mse':
                W = (model.u_encoder.weight.T @ model.v_encoder.weight).double()
                r = self.v_bank - self.u_bank @ W
                return gaussian_samples(self.u @ W, r.T @ r / r.shape[0], self.M), None
            score = clip_score_fn(model, self.loss_cfg.get('inner_product', True),
                                  self.loss_cfg.get('normalize', False))
            return sir_samples(score, self.u, self.v_bank.float(), self.M)

    def __call__(self, model):
        """Flat dict of metrics for logging (prefix 'bb/')."""
        was_training = model.training
        model.eval()
        samples, ess = self.model_samples(model)
        model.train(was_training)
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
        return out
