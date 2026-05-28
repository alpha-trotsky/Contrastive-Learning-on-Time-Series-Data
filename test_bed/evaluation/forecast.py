# evaluation/forecast.py

import torch


@torch.no_grad()
def forecast_mse(model, modality, n_samples=500):
    """MSE between the model's u-from-v prediction and the analytical conditional mean.

    Given the learned cross-term A^ = logit_scale * W_u^T W_v, the model
    predicts u as  u_pred = v @ A^T.  We compare this against the MMSE
    estimator  u_cond = v @ (C_vv^{-1} C_uv^T)^T.

    Parameters
    ----------
    model     : LinearCLIP
    modality  : PairSampler with C_uu / C_vv / C_uv implemented
    n_samples : int — number of fresh samples to evaluate on

    Returns
    -------
    scalar tensor — mean squared error
    """
    _, v = modality.sample_pair(n_samples)
    v_f = v.float()

    A_hat = model.cross_term().float()              # (dim_u, dim_v)
    u_pred = v_f @ A_hat.T                          # (N, dim_u)

    C_uv = modality.C_uv().float()                 # (dim_u, dim_v)
    C_vv = modality.C_vv().float()                 # (dim_v, dim_v)
    K = torch.linalg.solve(C_vv, C_uv.T)           # (dim_v, dim_u)
    u_cond = v_f @ K                               # (N, dim_u)

    return torch.mean((u_pred - u_cond) ** 2)
