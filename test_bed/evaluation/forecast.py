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


# ---------------------------------------------------------------------------
# Model conditional means  E_nu[v | u]
#
# Each contrastive model ν tilts the product of the data marginals
# µ_u = N(0, C_uu), µ_v = N(0, C_vv).  Which tilting we use fixes the closed
# form of the conditional ν(v|u), and therefore the model's point prediction
# of v from u.  All three below return v_pred for a batch u_f of shape (N, du).
#
# Throughout:  A = model.cross_term() = s * W_u^T W_v   (s = logit_scale.exp())
# is the paper's cross-term, with the temperature already folded in.
# ---------------------------------------------------------------------------

@torch.no_grad()
def model_mean_cosine(model, u_f, modality):
    """Cosine tilting, model form (39): ν ∝ exp(<Gu, Hv>) µ_u µ_v.

    Paper eq (41b):  ν(v|u) = N( C_vv A^T u ,  C_vv ),  so
        E_nu[v|u] = C_vv A^T u   ->   batch form  V_pred = U A C_vv.

    Used for every cosine-tilting run (inner_product=True): the two-sided /
    one-sided conditional loss AND the joint loss share this model form.
    """
    A_hat = model.cross_term().float()                       # (du, dv), incl. s
    C_vv = modality.C_vv().float().to(u_f.device)            # (dv, dv)
    return u_f @ A_hat @ C_vv                                # (N, dv)


@torch.no_grad()
def model_mean_quadratic(model, u_f, modality):
    """L2 tilting, model form (46): ν ∝ exp(-0.5|Gu - Hv|^2) µ_u µ_v.

    Paper eq (48b):  ν(v|u) = N( (C + C_vv^-1)^-1 A^T u , (C + C_vv^-1)^-1 ), so
        E_nu[v|u] = (C + C_vv^-1)^-1 A^T u
                  ->  batch form  V_pred = U A (C + C_vv^-1)^-1.

    The logit_scale s multiplies the whole quadratic, i.e. G = sqrt(s) W_u,
    H = sqrt(s) W_v, hence  A = s W_u^T W_v (= cross_term)  and
    C = H^T H = s W_v^T W_v.  Used for every L2-tilting run (inner_product=False).
    """
    device = u_f.device
    s = model.logit_scale.exp().float()
    A_hat = model.cross_term().float()                       # s W_u^T W_v  (du, dv)
    Wv = model.v_encoder.weight.float()                      # (embed, dv)
    C_hat = s * (Wv.T @ Wv)                                  # s W_v^T W_v  (dv, dv)
    C_vv = modality.C_vv().float().to(device)                # (dv, dv)
    M = C_hat + torch.linalg.inv(C_vv)                       # C + C_vv^-1  (dv, dv)
    return u_f @ A_hat @ torch.linalg.inv(M)                 # (N, dv)


@torch.no_grad()
def model_mean_mse(model, u_f, modality):
    """MSE baseline: a plain linear regression head, no tilting.

    The MSE loss minimises ||u W_u^T W_v - v||^2, so the prediction is the
    least-squares map itself, with NO temperature and NO C_vv factor:
        V_pred = U (W_u^T W_v).
    (logit_scale never enters the MSE loss, so it stays frozen and must be
    excluded here -- that is why we use the raw weights, not cross_term().)
    """
    Wu = model.u_encoder.weight.float()                      # (embed, du)
    Wv = model.v_encoder.weight.float()                      # (embed, dv)
    return u_f @ (Wu.T @ Wv)                                 # (N, dv)


_MEAN_FNS = {
    'cosine':    model_mean_cosine,
    'quadratic': model_mean_quadratic,
    'mse':       model_mean_mse,
}


@torch.no_grad()
def predict_v(model, u_f, modality, mean_family):
    """Dispatch to the model conditional mean E_nu[v|u] for the given family."""
    try:
        return _MEAN_FNS[mean_family](model, u_f, modality)
    except KeyError:
        raise ValueError(f"Unknown mean_family: {mean_family!r}")


# ---------------------------------------------------------------------------
# Model conditional covariances  Cov_nu[v | u]
#
# For jointly-Gaussian ν the conditional covariance is independent of the
# conditioning value u, so each of these is a single closed-form matrix read
# straight off the encoders + data covariances -- no sampling, no residuals.
# The contrast between the two tilts is the whole point of the paper:
#   * cosine tilt is LINEAR in v -> only shifts the mean, never touches the
#     precision, so the covariance is stuck at the prior C_vv (Corollary 5.2).
#   * L2 tilt adds a -½ vᵀC v quadratic term -> adds C to the precision, so the
#     covariance can shrink and can match the truth (Corollary 5.4).
# ---------------------------------------------------------------------------

@torch.no_grad()
def model_cov_cosine(model, modality):
    """Cosine tilting, eq (41b): Cov_nu[v|u] = C_vv  (the prior; A drops out)."""
    device = next(model.parameters()).device
    return modality.C_vv().float().to(device)


@torch.no_grad()
def model_cov_quadratic(model, modality):
    """L2 tilting, eq (48b): Cov_nu[v|u] = (C + C_vv^-1)^-1,  C = s W_v^T W_v."""
    device = next(model.parameters()).device
    s = model.logit_scale.exp().float()
    Wv = model.v_encoder.weight.float()                      # (embed, dv)
    C_hat = s * (Wv.T @ Wv)                                  # s W_v^T W_v  (dv, dv)
    C_vv = modality.C_vv().float().to(device)                # (dv, dv)
    return torch.linalg.inv(C_hat + torch.linalg.inv(C_vv))  # (dv, dv)


@torch.no_grad()
def model_cov_mse(model, modality):
    """MSE baseline is a bare point predictor: no probabilistic model, so there
    is no encoder-derived conditional covariance to read off.  Returns None."""
    return None


_COV_FNS = {
    'cosine':    model_cov_cosine,
    'quadratic': model_cov_quadratic,
    'mse':       model_cov_mse,
}


@torch.no_grad()
def model_cov(model, modality, mean_family):
    """Dispatch to the model conditional covariance Cov_nu[v|u]; None for mse."""
    try:
        return _COV_FNS[mean_family](model, modality)
    except KeyError:
        raise ValueError(f"Unknown mean_family: {mean_family!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def forecast_mse(model, modality, mean_family, n_samples=500):
    """MSE between the model's v-from-u prediction and the true conditional mean.

    The model prediction is the conditional mean E_nu[v|u] of whichever model
    form was trained (cosine / quadratic / mse), computed by `predict_v`.
    The target is the data MMSE estimator  E_µ[v|u] = u K,  K = C_uu^-1 C_uv.

    With the model-appropriate head, this -> 0 exactly when the model recovers
    the true conditional mean (e.g. the cosine conditional loss at its optimum
    A* = C_uu^-1 C_uv C_vv^-1, where U A* C_vv = U K).

    Parameters
    ----------
    model       : LinearCLIP
    modality    : PairSampler with C_uu / C_vv / C_uv implemented
    mean_family : 'cosine' | 'quadratic' | 'mse'
    n_samples   : int -- number of fresh samples to evaluate on
    """
    device = next(model.parameters()).device
    u, _ = modality.sample_pair(n_samples)
    u_f = u.float().to(device)

    v_pred = predict_v(model, u_f, modality, mean_family)    # (N, dv)

    C_uu = modality.C_uu().float().to(device)               # (du, du)
    C_uv = modality.C_uv().float().to(device)               # (du, dv)
    K = torch.linalg.solve(C_uu, C_uv)                      # C_uu^-1 C_uv  (du, dv)
    v_cond = u_f @ K                                         # true E[v|u]  (N, dv)

    return torch.mean((v_pred - v_cond) ** 2)


@torch.no_grad()
def residual_cov_error(model, modality, mean_family, n_samples=1000):
    """Relative Frobenius error between residual covariance and C_{v|u}.

    Residual = v - E_nu[v|u] (using the model-appropriate head).  When the
    point prediction recovers the true conditional mean, the residual
    covariance equals the irreducible data conditional covariance
        C_{v|u} = C_vv - C_uv^T C_uu^-1 C_uv,
    so this -> 0.  (Note: this measures the quality of the *mean* recovery via
    the residual; it is not the model's generative covariance, which for the
    cosine conditional loss is the larger C_vv -- see Corollary 5.2.)

    Returns  ||Cov_residual - C_{v|u}|| / ||C_{v|u}||.
    """
    device = next(model.parameters()).device
    u, v = modality.sample_pair(n_samples)
    u_f, v_f = u.float().to(device), v.float().to(device)

    v_pred = predict_v(model, u_f, modality, mean_family)
    residuals = v_f - v_pred                                # (N, dv)
    cov_residual = residuals.T @ residuals / n_samples      # (dv, dv)

    C_uu = modality.C_uu().float().to(device)
    C_vv = modality.C_vv().float().to(device)
    C_uv = modality.C_uv().float().to(device)
    K = torch.linalg.solve(C_uu, C_uv)
    cov_theory = C_vv - C_uv.T @ K                          # C_{v|u}  (dv, dv)

    diff = torch.linalg.norm(cov_residual - cov_theory, ord='fro')
    return diff / torch.linalg.norm(cov_theory, ord='fro')


@torch.no_grad()
def model_cov_error(model, modality, mean_family):
    """Relative Frobenius error between the model's OWN closed-form conditional
    covariance Cov_nu[v|u] (read off the encoders) and the true data
    conditional covariance C_{v|u} = C_vv - C_uv^T C_uu^-1 C_uv.

    Unlike `residual_cov_error` (which estimates spread from residuals), this is
    purely analytic: it asks "what variance does the model CLAIM, and is it
    right?".  Expected behaviour:
      * cosine    -> a fixed nonzero floor (claims C_vv, always too big).
      * quadratic -> can -> 0 (one-sided L2 matches C_{v|u}, Corollary 5.4).
      * mse       -> None (point predictor has no claimed covariance).

    Returns  ||Cov_model - C_{v|u}|| / ||C_{v|u}||,  or None for mse.
    """
    cov_model = model_cov(model, modality, mean_family)
    if cov_model is None:
        return None

    device = next(model.parameters()).device
    C_uu = modality.C_uu().float().to(device)
    C_vv = modality.C_vv().float().to(device)
    C_uv = modality.C_uv().float().to(device)
    K = torch.linalg.solve(C_uu, C_uv)
    cov_theory = C_vv - C_uv.T @ K                          # C_{v|u}  (dv, dv)

    diff = torch.linalg.norm(cov_model - cov_theory, ord='fro')
    return diff / torch.linalg.norm(cov_theory, ord='fro')
