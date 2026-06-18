# evaluation/retrieval.py
"""Post-training evaluation on a FIXED test batch.

Two families of metrics, both averaged over many queries (so they are stable,
unlike the single-sample in-run similarity):

  * averaged predicted-vs-true similarity + margin against shuffled futures
    (the stable version of the old self_sim / rand_sim);
  * cross-modal retrieval recall@k in embedding space -- the downstream task
    the contrastive loss actually optimises.

The eval batch is drawn from a fixed seed, so every model is scored on the
SAME data and the numbers are directly comparable across loss variants.
"""

import torch


def _cos_rows(A, B, eps=1e-12):
    """Row-wise cosine similarity between matched rows of A and B.  -> (N,)"""
    num = (A * B).sum(dim=1)
    den = A.norm(dim=1).clamp_min(eps) * B.norm(dim=1).clamp_min(eps)
    return num / den


def _recall_at_k(S, ks):
    """Score matrix S (N_query x N_cand) with the true match on the diagonal.

    Returns {k: recall@k} = fraction of queries whose true match lands in the
    top-k highest-scoring candidates.
    """
    N = S.shape[0]
    targets = torch.arange(N, device=S.device)
    order = S.argsort(dim=1, descending=True)                 # best candidate first
    rank = (order == targets.unsqueeze(1)).float().argmax(dim=1)   # 0-based rank of truth
    return {k: (rank < k).float().mean().item() for k in ks}


@torch.no_grad()
def evaluate_encoders(model, modality, n_eval=512, seed=1234, ks=(1, 5, 10)):
    """Fixed-batch eval.  Returns a flat dict of scalar metrics (wandb-ready)."""
    device = next(model.parameters()).device
    torch.manual_seed(seed)                                   # identical batch for every model
    u, v = modality.sample_pair(n_eval)
    u_f = u.float().to(device)
    v_f = v.float().to(device)

    g_u = model.encode_u(u_f)                                 # (N, embed)
    g_v = model.encode_v(v_f)                                 # (N, embed)

    # ---- cross-modal retrieval (embedding-space cosine) ----
    gu_n = g_u / g_u.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gv_n = g_v / g_v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    S = gu_n @ gv_n.T                                         # row i = query u_i vs candidates v_j
    rec_u2v = _recall_at_k(S, ks)                            # past  -> future
    rec_v2u = _recall_at_k(S.T, ks)                          # future -> past

    # ---- averaged predicted-vs-true similarity (stable self_sim) ----
    H = model.v_encoder.weight                                # (embed, dv)
    v_pred = g_u @ H                                          # (N, dv): score direction in v-space
    self_sim = _cos_rows(v_pred, v_f)                         # (N,)
    perm = torch.randperm(n_eval, device=device)
    rand_sim = _cos_rows(v_pred, v_f[perm])                   # (N,): shuffled (unrelated) futures

    out = {
        'eval/self_sim_mean': self_sim.mean().item(),
        'eval/self_sim_std':  self_sim.std().item(),
        'eval/rand_sim_mean': rand_sim.mean().item(),
        'eval/margin':        (self_sim.mean() - rand_sim.mean()).item(),
    }
    for k in ks:
        out[f'eval/recall@{k}_u2v'] = rec_u2v[k]
        out[f'eval/recall@{k}_v2u'] = rec_v2u[k]
    return out
