"""Validate the black-box scoring pipeline on the Gaussian test bed, where every
model's predictive distribution is also known in closed form.

Checks:
  * SIR samples from a model's score reproduce that model's closed-form nu(v|u)
    (same scores up to Monte-Carlo error)  -> the black-box adapter is right.
  * The scores rank  oracle < L2 < cosine (over-dispersed) < climatology, and
    coverage shows cosine over-covering, as Cor 5.2 / 5.4 predict.
  * The old cosine self-similarity, for contrast.

Usage (from test_bed/):  python -m analysis.validate_scoring
"""
import json
from pathlib import Path

import torch
import yaml

from experiment import build_signal, build_modality, build_model
from evaluation.forecast import predict_v, model_cov
from evaluation.predictors import (gaussian_samples, point_samples, climatology_samples,
                                   sir_samples, clip_score_fn)
from evaluation.scoring import score_forecast, add_skills

MODELS = [
    # (label, weights, loss cfg)
    ('cosine_cond_K256', 'results/batch_sweep/cosine_K256.pt',
     {'type': 'conditional', 'inner_product': True}),
    ('l2_v|u_K64', 'results/batch_sweep/l2_vgu_K64.pt',
     {'type': 'one_sided_u', 'inner_product': False}),
    ('l2_u|v_saved20k', 'results/fill_one_sided_v_l2/model.pt',
     {'type': 'one_sided_v', 'inner_product': False}),
    ('mse_saved20k', 'results/fill_mse/model.pt', {'type': 'mse'}),
]


def main(N=1000, M=500, B=50000, seed=0):
    torch.manual_seed(seed)
    cfg = yaml.safe_load(open('config.yaml'))
    modality = build_modality(cfg['modality'], build_signal(cfg['signal']))
    C_uu, C_vv, C_uv = modality.C_uu(), modality.C_vv(), modality.C_uv()
    K = torch.linalg.solve(C_uu, C_uv)
    C_vgu = C_vv - C_uv.T @ K

    bank_u, bank = modality.sample_pair(B)           # "training" pairs
    u, y = modality.sample_pair(N)                   # test pairs
    S = {}
    S['oracle'] = gaussian_samples(u @ K, C_vgu, M)
    S['climatology'] = climatology_samples(bank, N, M)
    ess = {}
    old_sim = {}

    for label, path, loss in MODELS:
        if not Path(path).exists():
            print(f'skip {label}: {path} missing')
            continue
        model = build_model(cfg['model'], modality)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        if loss['type'] == 'mse':
            pred = predict_v(model, u.float(), modality, 'mse').double()
            S[f'{label} point'] = point_samples(pred)
            # homoscedastic Gaussian on training residuals: a fair probabilistic MSE baseline
            r = bank - predict_v(model, bank_u.float(), modality, 'mse').double()
            S[f'{label} +gauss-resid'] = gaussian_samples(pred, r.T @ r / B, M)
        else:
            fam = 'cosine' if loss['inner_product'] else 'quadratic'
            mean = predict_v(model, u.float(), modality, fam).double()
            cov = model_cov(model, modality, fam).double()
            S[f'{label} closed-form'] = gaussian_samples(mean, cov, M)
            S[f'{label} SIR'], e = sir_samples(clip_score_fn(model, loss['inner_product']),
                                               u, bank.float(), M)
            ess[label] = e.median().item()
        # the old metric: cosine(u_features @ W_v, v)
        with torch.no_grad():
            vp = model.encode_u(u.float()) @ model.v_encoder.weight
            old_sim[label] = torch.nn.functional.cosine_similarity(vp, y.float(), dim=1).mean().item()

    table = add_skills({k: score_forecast(s, y) for k, s in S.items()})
    cols = ['energy', 'energy_excess', 'crps', 'variogram', 'mse_mean', 'cov50', 'cov90', 'width90', 'cal_err']
    print(f'{"forecaster":34s}' + ''.join(f'{c:>14s}' for c in cols))
    for k, r in table.items():
        print(f'{k:34s}' + ''.join(f'{r.get(c, float("nan")):14.4f}' for c in cols))
    print('\nmedian SIR ESS (bank %d):' % B, {k: round(v) for k, v in ess.items()})
    print('old cosine self-sim:', {k: round(v, 3) for k, v in old_sim.items()})
    Path('results/scoring').mkdir(parents=True, exist_ok=True)
    json.dump({'table': table, 'ess': ess, 'old_self_sim': old_sim},
              open('results/scoring/gaussian_validation.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
