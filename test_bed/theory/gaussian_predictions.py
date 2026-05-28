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