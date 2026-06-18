import torch


def matrix_sqrt(mat):
    eigvals, eigvecs = torch.linalg.eigh(mat)
    return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T

def matrix_inv_sqrt(mat):
    eigvals, eigvecs = torch.linalg.eigh(mat)
    return eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals)) @ eigvecs.T

def predicted_A_conditional(C_uu, C_uv, C_vv):
    # Theorem 5.1: A* = C_uu^{-1} C_uv C_vv^{-1}
    return torch.linalg.solve(C_uu, C_uv) @ torch.linalg.inv(C_vv)

def predicted_A_joint(C_uu, C_uv, C_vv):
    # Theorem 5.6: SVD-based formula with h function
    # h(sigma) = (1/sigma) * (0.5*sqrt(1 + 4*sigma^2) - 0.5)
    M = torch.linalg.solve(matrix_sqrt(C_uu), C_uv)
    M = M @ torch.linalg.inv(matrix_sqrt(C_vv))
    U, S, Vt = torch.linalg.svd(M)
    h_S = (1/S) * (0.5 * torch.sqrt(1 + 4*S**2) - 0.5)
    return matrix_inv_sqrt(C_uu) @ U @ torch.diag(h_S) @ Vt @ matrix_inv_sqrt(C_vv)

def predicted_A_quadratic_v(C_uu, C_uv, C_vv):
    # Theorem 5.3, v|u one-sided (lambda=(0,2), our `one_sided_v_l2`):
    #   A* = C_uu^{-1} C_uv C_{v|u}^{-1},  C_{v|u} = C_vv - C_uv^T C_uu^{-1} C_uv
    # Same shape as the cosine conditional target, but the right-hand factor is
    # the *conditional* precision C_{v|u}^{-1} instead of the marginal C_vv^{-1}.
    C_vgu = C_vv - C_uv.T @ torch.linalg.solve(C_uu, C_uv)
    return torch.linalg.solve(C_uu, C_uv) @ torch.linalg.inv(C_vgu)

def predicted_A_quadratic_u(C_uu, C_uv, C_vv):
    # Theorem 5.3, u|v one-sided (lambda=(2,0), the paper's `one_sided_u`):
    #   A* = C_{u|v}^{-1} C_uv C_vv^{-1},  C_{u|v} = C_uu - C_uv C_vv^{-1} C_uv^T
    C_ugv = C_uu - C_uv @ torch.linalg.solve(C_vv, C_uv.T)
    return torch.linalg.solve(C_ugv, C_uv) @ torch.linalg.inv(C_vv)