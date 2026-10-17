"""Goal 1/2 sweep: train a model on one non-Gaussian domain and score it against
that domain's perfect forecaster (oracle).

One run = (domain, dial, model, encoder, seed).  Models: the {cosine, L2} x
{one-sided, two-sided, joint} contrastive grid, normalised cosine (real CLIP),
MSE regression, and a heteroscedastic Gaussian regressor (mean + per-step
variance, Gaussian NLL).  --encoder picks linear or MLP networks.  Contrastive
models are scored through their own score function only (SIR over a bank of
training futures); MSE with Gaussian residual noise; the Gaussian regressor by
sampling its own predictive.

Usage (from test_bed/):
    python -m analysis.domain_sweep --domain t --dial 3 --model cosine --encoder mlp
"""
import argparse
import json
import math
import zlib
from pathlib import Path

import torch
import yaml

from experiment import build_signal, build_modality, build_model, build_loss, resolve_theory_target
from modalities.nonlinear import (StudentTPastFuture, WarpedPastFuture, DoubleWellPastFuture,
                                  _gaussian_conditional)
from models.mlp_regressor import MeanRegressor, GaussianRegressor
from evaluation.predictors import (gaussian_samples, point_samples, climatology_samples,
                                   sir_samples, clip_score_fn)
from evaluation.scoring import score_forecast, add_skills
from evaluation.theory_match import theory_match_whitened_error

LOSSES = {
    'cosine': {'type': 'conditional', 'inner_product': True, 'lambda_u': 0.5, 'lambda_v': 0.5},
    'l2':     {'type': 'one_sided_u', 'inner_product': False, 'lambda_u': 1.0, 'lambda_v': 0.0},
    'mse':    {'type': 'mse'},
    # completing the grid: {cosine, L2} x {one-sided past->future, two-sided, joint}
    'cosine_1s':    {'type': 'one_sided_u', 'inner_product': True, 'lambda_u': 1.0, 'lambda_v': 0.0},
    'cosine_joint': {'type': 'joint', 'inner_product': True},
    'l2_2s':        {'type': 'conditional', 'inner_product': False, 'lambda_u': 0.5, 'lambda_v': 0.5},
    'l2_joint':     {'type': 'joint', 'inner_product': False},
    # real CLIP: two-sided, cosine on L2-normalised embeddings
    'ncos':         {'type': 'conditional', 'inner_product': True, 'normalize': True,
                     'lambda_u': 0.5, 'lambda_v': 0.5},
    # heteroscedastic Gaussian regressor (mean + per-step log-variance, Gaussian NLL)
    'gauss':        {'type': 'gauss'},
}
REGRESSORS = ('mse', 'gauss')


def build_domain(name, dial, cfg):
    if name == 'dwell':
        return DoubleWellPastFuture(dial, n_particles=1000)
    base = build_modality(cfg['modality'], build_signal(cfg['signal']))   # config.yaml, unchanged
    if name == 't':
        return StudentTPastFuture(base, dial)
    if name == 'warp':
        return WarpedPastFuture(base, dial)
    raise ValueError(name)


def make_data(mod, n, chunk=10000):
    us, vs = zip(*(mod.sample_pair(min(chunk, n - i)) for i in range(0, n, chunk)))
    return torch.cat(us), torch.cat(vs)


def cond_coverage(samples, y, spread, level=0.9, bins=3):
    """90% coverage within terciles of the oracle's predicted spread
    (low / mid / high uncertainty cases).  Marginal coverage can hide a model
    that is too wide on easy cases and too narrow on hard ones."""
    q = torch.tensor([(1 - level) / 2, (1 + level) / 2], dtype=samples.dtype)
    lo, hi = torch.quantile(samples, q, dim=1)
    per_query = ((y >= lo) & (y <= hi)).double().mean(1)
    edges = torch.quantile(spread, torch.linspace(0, 1, bins + 1, dtype=spread.dtype))
    out = []
    for b in range(bins):
        m = (spread >= edges[b]) & (spread <= edges[b + 1])
        out.append(per_query[m].mean().item())
    return out


def r2(X_tr, Y_tr, X_te, Y_te):
    """Held-out R^2 of a least-squares fit Y ~ X (with intercept)."""
    one = lambda X: torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], 1)
    B = torch.linalg.lstsq(one(X_tr), Y_tr).solution
    res = Y_te - one(X_te) @ B
    return 1 - (res ** 2).sum().item() / ((Y_te - Y_tr.mean(0)) ** 2).sum().item()


@torch.no_grad()
def latent_recovery(model, mod, n=6000, n_fit=5000):
    """Bent GP: does the u-encoder 'un-bend' the data?  Linear-regress the true
    latent quantities on the learned u-features (held out) and compare with the
    same regression on the raw past.  Targets: the latent past x_u itself, and
    the latent best forecast x_u K (the part of the past that matters)."""
    u, _ = mod.sample_pair(n)
    xu = mod.g_inv(u)
    K, _ = _gaussian_conditional(mod.base.C_uu(), mod.base.C_vv(), mod.base.C_uv())
    feats = model.encode_u(u.float()).double()
    out = {}
    for name, Y in (('latent_past', xu), ('latent_forecast', xu @ K)):
        out[f'r2_{name}_features'] = r2(feats[:n_fit], Y[:n_fit], feats[n_fit:], Y[n_fit:])
        out[f'r2_{name}_raw'] = r2(u[:n_fit], Y[:n_fit], u[n_fit:], Y[n_fit:])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', choices=['t', 'warp', 'dwell'], required=True)
    p.add_argument('--dial', type=float, required=True)
    p.add_argument('--model', choices=LOSSES, required=True)
    p.add_argument('--encoder', choices=['linear', 'mlp'], default='linear')
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--steps', type=int, default=15000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--n_train', type=int, default=200000)
    p.add_argument('--n_test', type=int, default=500)
    p.add_argument('--M', type=int, default=300)
    p.add_argument('--bank', type=int, default=50000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='results/domain_sweep')
    a = p.parse_args()
    torch.set_num_threads(1)

    cfg = yaml.safe_load(open('config.yaml'))
    cfg['loss'] = LOSSES[a.model]
    # same data for every model at a given (domain, dial, seed)
    torch.manual_seed(1000 * a.seed + zlib.crc32(f'{a.domain}{a.dial}'.encode()) % 1000)
    mod = build_domain(a.domain, a.dial, cfg)
    u_tr, v_tr = make_data(mod, a.n_train)
    u_te, y = make_data(mod, a.n_test)
    oracle = mod.oracle_samples(u_te, a.M)
    spread = mod.oracle_spread(u_te)

    torch.manual_seed(a.seed)
    if a.encoder == 'mlp':
        cfg['model'].update(type='mlp_clip', hidden=a.hidden, depth=a.depth, skip=True)
    regressor = a.model in REGRESSORS and not (a.model == 'mse' and a.encoder == 'linear')
    if regressor:
        depth = a.depth if a.encoder == 'mlp' else 0
        cls = MeanRegressor if a.model == 'mse' else GaussianRegressor
        model = cls(mod.dim_u, mod.dim_v, a.hidden, depth)
        loss_fn = None
    else:
        model = build_model(cfg['model'], mod)
        loss_fn = build_loss(cfg['loss'], model)
    if hasattr(model, 'set_input_stats'):
        model.set_input_stats(u_tr.float(), v_tr.float())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    for step in range(a.steps):
        idx = torch.randint(0, a.n_train, (a.batch,))
        u, v = u_tr[idx], v_tr[idx]
        if regressor:
            loss = model.loss(u, v)
        else:
            gu, gv, s = model(u.float(), v.float())
            loss = loss_fn(gu, gv, s, u, v)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    bank = v_tr[:a.bank]
    S = {'oracle': oracle, 'climatology': climatology_samples(bank, a.n_test, a.M)}
    extra = {'final_loss': loss.item()}
    if not regressor:
        extra['logit_scale'] = model.logit_scale.exp().item()
    if a.model == 'mse' and regressor:
        with torch.no_grad():
            pred = model.predict(u_te).double()
            r = v_tr[:a.bank] - model.predict(u_tr[:a.bank]).double()
        S['model_point'] = point_samples(pred)
        S['model'] = gaussian_samples(pred, r.T @ r / a.bank, a.M)
    elif a.model == 'gauss':
        with torch.no_grad():
            mu, sd = (x.double() for x in model.predict(u_te))
            mu_b, sd_b = (x.double() for x in model.predict(u_tr[:a.bank]))
        z = (v_tr[:a.bank] - mu_b) / sd_b          # standardised residuals
        R = torch.corrcoef(z.T)                    # residual correlation across steps
        S['model'] = mu[:, None, :] + sd[:, None, :] * gaussian_samples(torch.zeros_like(mu), R, a.M)
        S['model_point'] = point_samples(mu)
    elif a.model == 'mse':
        W = (model.u_encoder.weight.T @ model.v_encoder.weight).double().detach()
        pred = u_te @ W
        S['model_point'] = point_samples(pred)
        r = v_tr[:a.bank] - u_tr[:a.bank] @ W
        S['model'] = gaussian_samples(pred, r.T @ r / a.bank, a.M)
    else:
        L = LOSSES[a.model]
        S['model'], ess = sir_samples(clip_score_fn(model, L['inner_product'], L.get('normalize', False)),
                                      u_te, bank.float(), a.M)
        extra['ess_median'] = ess.median().item()
        extra['ess_min'] = ess.min().item()
        extra['ess_p05'] = ess.quantile(0.05).item()
        if a.domain == 'warp':
            extra.update(latent_recovery(model, mod))
    if a.domain == 't' and a.encoder == 'linear' and not regressor and not LOSSES[a.model].get('normalize'):
        # same covariances as the GP, so the Gaussian theory's A* still applies
        target = resolve_theory_target(cfg['loss'])
        w = theory_match_whitened_error(model, mod, target)
        extra['whitened_theory_err'] = None if w is None else w.item()

    table = add_skills({k: score_forecast(s, y) for k, s in S.items()})
    for k in ('oracle', 'model'):
        table[k]['cov90_by_spread'] = cond_coverage(S[k].double(), y.double(), spread.double())

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # linear runs keep the goal-1 file names
    enc = '' if a.encoder == 'linear' else f'_{a.encoder}'
    tag = f'{a.domain}_{a.dial:g}_{a.model}{enc}_s{a.seed}'
    json.dump({'args': vars(a), 'scores': table, 'extra': extra}, open(out / f'{tag}.json', 'w'), indent=1)
    m, o = table['model'], table['oracle']
    print(f"{tag}: energy model {m['energy']:.4f} oracle {o['energy']:.4f} clim {table['climatology']['energy']:.4f} "
          f"| cov90 {m['cov90']:.3f} by-spread {[round(x, 2) for x in m['cov90_by_spread']]} | {extra}")


if __name__ == '__main__':
    main()
