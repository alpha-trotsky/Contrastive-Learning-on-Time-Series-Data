"""Follow-ups from issue #3 ("order of attack" items 1-3).

  (1) E_dir / scale on the saved cosine weights  -> is the 0.62 floor direction or scale?
  (2) SVD of the whitened cross-covariance        -> canonical correlations, MI, h(sigma)
  (3) Retrieval: embedding-cosine (old eval) vs the model's OWN logit vs the
      Bayes-optimal critic (true PMI) on the same fixed 512-way batch.

Usage (from test_bed/):  python -m analysis.issue3_followups
"""
import math
import yaml
import torch

from experiment import build_signal, build_modality, build_model, resolve_theory_target
from evaluation.theory_match import theory_match_error, theory_match_dir_error
from evaluation.retrieval import _recall_at_k
from theory.gaussian_predictions import matrix_inv_sqrt

RUNS = {
    'phase1_baseline (cosine cond, embed=20)': ('../results/phase1_gp_conditional_baseline', '../results/phase1_gp_conditional_baseline/model.pt'),
    'one_sided_v_dot (cosine, 20k)':           ('results/fill_one_sided_v_dot', 'results/fill_one_sided_v_dot/model.pt'),
    'one_sided_v_l2  (L2, 20k)':               ('results/fill_one_sided_v_l2', 'results/fill_one_sided_v_l2/model.pt'),
    'mse (20k)':                                ('results/fill_mse', 'results/fill_mse/model.pt'),
}


def load(cfg_dir, model_path):
    with open(f'{cfg_dir}/config.yaml') as f:
        cfg = yaml.safe_load(f)
    gen = build_signal(cfg['signal'])
    modality = build_modality(cfg['modality'], gen)
    model = build_model(cfg['model'], modality)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return cfg, modality, model


def spectrum(modality):
    C_uu, C_vv, C_uv = modality.C_uu(), modality.C_vv(), modality.C_uv()
    M = matrix_inv_sqrt(C_uu) @ C_uv @ matrix_inv_sqrt(C_vv)
    S = torch.linalg.svdvals(M)
    mi = -0.5 * torch.log1p(-S.clamp(max=1 - 1e-12) ** 2).sum()
    h = (0.5 * torch.sqrt(1 + 4 * S ** 2) - 0.5) / S
    return S, mi, h


def gaussian_pmi_scores(modality, u, v):
    """S[i, j] = log p(v_j | u_i) - log p(v_j)  (Bayes-optimal u->v retrieval critic)."""
    C_uu, C_vv, C_uv = modality.C_uu(), modality.C_vv(), modality.C_uv()
    K = torch.linalg.solve(C_uu, C_uv)
    C_vgu = C_vv - C_uv.T @ K
    Pc, Pm = torch.linalg.inv(C_vgu), torch.linalg.inv(C_vv)
    mu = u @ K                                                       # (N, dv)
    # -1/2 (v_j - mu_i)^T Pc (v_j - mu_i) + 1/2 v_j^T Pm v_j  (constants dropped)
    vPv = ((v @ Pc) * v).sum(1)
    mPm = ((mu @ Pc) * mu).sum(1)
    cross = mu @ Pc @ v.T
    return -0.5 * (mPm[:, None] - 2 * cross + vPv[None, :]) + 0.5 * ((v @ Pm) * v).sum(1)[None, :]


@torch.no_grad()
def model_logits(model, cfg, u, v):
    """The model's own score matrix (what the loss actually trains), u-rows x v-cols."""
    gu, gv = model.encode_u(u.float()), model.encode_v(v.float())
    s = model.logit_scale.exp()
    if cfg['loss']['type'] == 'mse':
        # regression head: rank candidates by -||v_pred - v_j||^2
        vp = u.float() @ (model.u_encoder.weight.T @ model.v_encoder.weight)
        return -torch.cdist(vp, v.float()) ** 2
    if cfg['loss'].get('inner_product', True):
        return s * gu @ gv.T
    return -0.5 * s * torch.cdist(gu, gv) ** 2


@torch.no_grad()
def cosine_embed_scores(model, u, v):
    gu = torch.nn.functional.normalize(model.encode_u(u.float()), dim=1)
    gv = torch.nn.functional.normalize(model.encode_v(v.float()), dim=1)
    return gu @ gv.T


def main():
    torch.set_grad_enabled(False)
    ks = (1, 5, 10)
    first = True
    for label, (cfg_dir, path) in RUNS.items():
        cfg, modality, model = load(cfg_dir, path)
        if first:
            S, mi, h = spectrum(modality)
            print('=== (2) canonical correlations of past(50) / future(50), sigma=0.05 ===')
            print('top-10 sigma :', [round(x, 4) for x in S[:10].tolist()])
            print('# sigma > 0.9 / 0.5 / 0.1 :', int((S > 0.9).sum()), int((S > 0.5).sum()), int((S > 0.1).sum()))
            print(f'I(u;v) = {mi.item():.2f} nats   (log 256 = {math.log(256):.2f}, log 512 = {math.log(512):.2f})')
            print('h(sigma) (joint-loss singular values) top-5:', [round(x, 3) for x in h[:5].tolist()])
            torch.manual_seed(1234)
            u_e, v_e = modality.sample_pair(512)
            bayes = gaussian_pmi_scores(modality, u_e.double(), v_e.double())
            rb = _recall_at_k(bayes, ks)
            print('\n=== (3) Bayes-optimal retrieval ceiling (true PMI critic), 512-way u->v ===')
            print('  ' + '  '.join(f'R@{k}={rb[k]:.3f}' for k in ks))
            print('\n=== (1)+(3) per saved model ===')
            first = False

        target = resolve_theory_target(cfg['loss'])
        err = theory_match_error(model, modality, target)
        e_dir, scale = theory_match_dir_error(model, modality, target)
        r_cos = _recall_at_k(cosine_embed_scores(model, u_e, v_e), ks)
        r_log = _recall_at_k(model_logits(model, cfg, u_e, v_e), ks)
        f = lambda x: float('nan') if x is None else x.item()
        print(f'{label:42s} steps={cfg["training"]["num_steps"]:6d} target={target:18s} '
              f'theory_err={f(err):.3f} E_dir={f(e_dir):.3f} scale={f(scale):.3f} '
              f's={model.logit_scale.exp().item():.2f}')
        print(f'{"":42s} R@1/5/10 embed-cosine = {r_cos[1]:.3f}/{r_cos[5]:.3f}/{r_cos[10]:.3f} | '
              f'own logit = {r_log[1]:.3f}/{r_log[5]:.3f}/{r_log[10]:.3f}')


if __name__ == '__main__':
    main()
