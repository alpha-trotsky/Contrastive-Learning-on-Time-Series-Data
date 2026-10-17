"""Does the cosine theory-error floor depend on the number of negatives K (= batch size)?

Hypothesis: the bilinear (cosine) critic u^T A v cannot represent the true PMI,
which carries a v-only quadratic term.  For a misspecified critic family the
minimiser of the K-sample softmax loss depends on K and only reaches the
population optimum A* (Thm 5.1) as K -> inf.  The L2 critic contains the PMI,
so ranking-NCE consistency makes its optimum K-independent.

Usage (from test_bed/):
    python -m analysis.batch_sweep --loss cosine --batch 256 --steps 30000 --out results/batch_sweep
"""
import argparse
import json
from pathlib import Path

import torch
import yaml

from experiment import build_signal, build_modality, build_model, build_loss, resolve_theory_target
from evaluation.theory_match import (theory_match_error, theory_match_dir_error,
                                     theory_match_whitened_error)

LOSSES = {
    # two-sided cosine conditional (standard CLIP)
    'cosine': {'type': 'conditional', 'inner_product': True, 'lambda_u': 0.5, 'lambda_v': 0.5},
    # one-sided L2 on p(v|u) only  (this code's 'one_sided_u': CE over logits_per_image)
    'l2_vgu': {'type': 'one_sided_u', 'inner_product': False, 'lambda_u': 1.0, 'lambda_v': 0.0},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--loss', choices=LOSSES, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='results/batch_sweep')
    a = p.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    cfg = yaml.safe_load(open('config.yaml'))
    cfg['loss'] = LOSSES[a.loss]
    modality = build_modality(cfg['modality'], build_signal(cfg['signal']))
    model = build_model(cfg['model'], modality)
    loss_fn = build_loss(cfg['loss'], model)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    target = resolve_theory_target(cfg['loss'])

    hist = []
    for step in range(a.steps + 1):
        u, v = modality.sample_pair(a.batch)
        gu, gv, s = model(u.float(), v.float())
        loss = loss_fn(gu, gv, s, u, v)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % a.every == 0:
            raw = theory_match_error(model, modality, target).item()
            d, sc = theory_match_dir_error(model, modality, target)
            w = theory_match_whitened_error(model, modality, target).item()
            row = dict(step=step, loss=loss.item(), raw=raw, dir=d.item(), scale=sc.item(),
                       whitened=w, s=s.item())
            hist.append(row)
            print(json.dumps(row), flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{a.loss}_K{a.batch}'
    json.dump({'args': vars(a), 'target': target, 'history': hist}, open(out / f'{tag}.json', 'w'))
    torch.save(model.state_dict(), out / f'{tag}.pt')


if __name__ == '__main__':
    main()
