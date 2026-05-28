# evaluation/theory_match.py

import torch
from theory.gaussian_predictions import predicted_A_conditional, predicted_A_joint


@torch.no_grad()
def theory_match_error(model, modality, loss_type='conditional'):
    """Relative Frobenius error between the learned cross-term and A*.

    Parameters
    ----------
    model     : LinearCLIP
    modality  : PairSampler with C_uu / C_vv / C_uv implemented
    loss_type : 'conditional' | 'joint'

    Returns
    -------
    scalar tensor — ||G^T H - A*|| / ||A*||
    """
    C_uu = modality.C_uu()
    C_vv = modality.C_vv()
    C_uv = modality.C_uv()
    if loss_type == 'conditional':
        A_star = predicted_A_conditional(C_uu, C_uv, C_vv)
    elif loss_type == 'joint':
        A_star = predicted_A_joint(C_uu, C_uv, C_vv)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")
    A_learned = model.cross_term().double()
    return torch.norm(A_learned - A_star, p='fro') / torch.norm(A_star, p='fro')
