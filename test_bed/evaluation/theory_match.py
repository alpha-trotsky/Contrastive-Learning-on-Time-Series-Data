# evaluation/theory_match.py

import torch
from theory.gaussian_predictions import (
    predicted_A_conditional,
    predicted_A_joint,
    predicted_A_quadratic_u,
    predicted_A_quadratic_v,
)


def _get_matrices(model, modality, target):
    """Return (A_learned, A_star) for the given target, or None if no closed form."""
    if target == 'none':
        return None

    device = next(model.parameters()).device
    C_uu = modality.C_uu().double().to(device)
    C_vv = modality.C_vv().double().to(device)
    C_uv = modality.C_uv().double().to(device)

    if target == 'mse':
        # MSE trains the RAW map W_u^T W_v -> K = C_uu^-1 C_uv (no logit_scale),
        # so compare the unscaled weights, not cross_term().
        A_star = torch.linalg.solve(C_uu, C_uv)
        A_learned = (model.u_encoder.weight.T @ model.v_encoder.weight).double()
    else:
        if target == 'cosine_conditional':
            A_star = predicted_A_conditional(C_uu, C_uv, C_vv)   # Thm 5.1
        elif target == 'cosine_joint':
            A_star = predicted_A_joint(C_uu, C_uv, C_vv)         # Thm 5.6
        elif target == 'quadratic_v':
            A_star = predicted_A_quadratic_v(C_uu, C_uv, C_vv)   # Thm 5.3 (v|u)
        elif target == 'quadratic_u':
            A_star = predicted_A_quadratic_u(C_uu, C_uv, C_vv)   # Thm 5.3 (u|v)
        else:
            raise ValueError(f"Unknown theory target: {target!r}")
        A_learned = model.cross_term().double()

    return A_learned, A_star


@torch.no_grad()
def theory_match_error(model, modality, target):
    """Relative Frobenius error ||A - A*|| / ||A*||, or None if no closed form."""
    result = _get_matrices(model, modality, target)
    if result is None:
        return None
    A_learned, A_star = result
    return torch.norm(A_learned - A_star, p='fro') / torch.norm(A_star, p='fro')


@torch.no_grad()
def theory_match_dir_error(model, modality, target):
    """Direction error E_dir = sin∠(A, A*) and scale ratio ||A|| / ||A*||.

    Splits the total Frobenius error into:
      - E_dir = sqrt(1 - <A, A*>_F^2 / (||A||_F^2 ||A*||_F^2))  in [0, 1]
        0 means A and A* point the same direction (only scale differs).
      - scale = ||A||_F / ||A*||_F
        1 means the magnitudes match exactly.

    Returns (E_dir, scale) or (None, None) if no closed form exists.
    """
    result = _get_matrices(model, modality, target)
    if result is None:
        return None, None
    A_learned, A_star = result

    norm_A = torch.norm(A_learned, p='fro')
    norm_Astar = torch.norm(A_star, p='fro')

    frob_inner = (A_learned * A_star).sum()
    # Clamp the cosine to [-1, 1] to guard against tiny numerical overflows
    cos2 = (frob_inner / (norm_A * norm_Astar)).clamp(-1.0, 1.0) ** 2
    E_dir = torch.sqrt((1.0 - cos2).clamp(min=0.0))

    scale = norm_A / norm_Astar
    return E_dir, scale
