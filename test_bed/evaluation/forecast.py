# evaluation/forecast.py

import torch

@torch.no_grad()
def debug_Ahat(model, modality):
    device = next(model.parameters()).device
    A = model.cross_term().float()
    K = torch.linalg.solve(modality.C_uu().float().to(device), modality.C_uv().float().to(device))
    return {
        "A_norm":      torch.linalg.norm(A).item(),       # is it moving at all?
        "A_minus_K":   torch.linalg.norm(A - K).item(),   # distance to MMSE target
        "A_entry_00":  A[0, 0].item(),                    # watch one number tick
        "logit_scale": float(model.logit_scale),          # frozen? saturating?
    }

@torch.no_grad()
def forecast_mse(model, modality, n_samples=500):
    """MSE between the model's v-from-u prediction and the analytical conditional mean.

    Given the learned cross-term A^ = logit_scale * W_u^T W_v, the model
    predicts v as  v_pred = u @ A^.  We compare this against the MMSE
    estimator  v_cond = u @ (C_uu^{-1} C_uv).

    Parameters
    ----------
    model     : LinearCLIP
    modality  : PairSampler with C_uu / C_vv / C_uv implemented
    n_samples : int — number of fresh samples to evaluate on

    Returns
    -------
    scalar tensor — mean squared error
    """
    device = next(model.parameters()).device
    u, _ = modality.sample_pair(n_samples)
    u_f = u.float().to(device)

    A_hat = model.cross_term().float()              # (dim_u, dim_v)
    v_pred = u_f @ A_hat                            # (N, dim_v)

    C_uu = modality.C_uu().float().to(device)       # (dim_u, dim_u)
    C_uv = modality.C_uv().float().to(device)       # (dim_u, dim_v)
    K = torch.linalg.solve(C_uu, C_uv)              # (dim_u, dim_v)
    v_cond = u_f @ K                                # (N, dim_v)

    return torch.mean((v_pred - v_cond) ** 2)


@torch.no_grad()
def residual_cov_error(model, modality, n_samples=1000):
    """Frobenius distance between empirical residual covariance and theoretical conditional variance.

    If the model perfectly learns E[v|u], residuals v - v_pred should have
    covariance equal to C_vv - C_uv^T @ C_uu^{-1} @ C_uv.

    Returns
    -------
    scalar tensor — relative Frobenius error ||Cov_residual - Cov_theory|| / ||Cov_theory||
    """
    device = next(model.parameters()).device
    u, v = modality.sample_pair(n_samples)
    u_f, v_f = u.float().to(device), v.float().to(device)

    A_hat = model.cross_term().float()
    v_pred = u_f @ A_hat
    residuals = v_f - v_pred                                        # (N, dim_v)
    cov_residual = residuals.T @ residuals / n_samples              # (dim_v, dim_v)

    C_uu = modality.C_uu().float().to(device)
    C_vv = modality.C_vv().float().to(device)
    C_uv = modality.C_uv().float().to(device)
    K = torch.linalg.solve(C_uu, C_uv)
    cov_theory = C_vv - C_uv.T @ K                                 # (dim_v, dim_v)

    diff = torch.linalg.norm(cov_residual - cov_theory, ord='fro')
    return diff / torch.linalg.norm(cov_theory, ord='fro')
