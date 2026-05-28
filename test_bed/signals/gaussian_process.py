import torch
from signals.base import SignalGenerator

class GaussianProcess1D(SignalGenerator):
    """1D GP with Neumann BCs via KL expansion:

        r(t) = sum_j sqrt(lambda_j) * xi_j * phi_j(t)
        lambda_j = (pi^2 * j^2 + tau^2)^{-alpha}
        xi_j     ~ N(0, 1)
        phi_j    = ortho-normalised cosine basis (IDCT-2)

    This class owns only the pure GP machinery: eigenspectrum, the
    coeff-to-field map, and noiseless sampling.  Modality-specific
    concerns (noise, u_index slicing, conditioning) live in
    modalities/base.py::FieldCoeffModality.

    Parameters
    ----------
    alpha     : float  — regularity of the covariance operator
    tau       : float  — inverse length-scale
    dim_true  : int    — KL truncation (grid size)
    zero_mean : bool   — if True, zero out the constant mode (j=0)
                         so the process has zero mean
    """

    def __init__(self, alpha, tau, dim_true, zero_mean=True):
        self.alpha = alpha
        self.tau = tau
        self.dim_true = dim_true
        self.zero_mean = zero_mean
        # Precompute once: Phi[n, k] maps coeff k -> spatial point n
        self._Phi = self._build_map()

    # ------------------------------------------------------------------
    # Grid and spectrum
    # ------------------------------------------------------------------
    def dim_true(self):
        """Grid of dim_true evenly-spaced interior points on [0, 1]."""
        d = self.dim_true
        return torch.linspace(1 / (2 * d), (d - 0.5) / d, d, dtype=torch.float64)

    def eigenvalues(self):
        """Eigenvalues lambda_j = (pi^2 * j^2 + tau^2)^{-alpha}."""
        k = torch.arange(self.dim_true, dtype=torch.float64)
        eigs = (torch.pi ** 2 * k ** 2 + self.tau ** 2) ** (-self.alpha)
        if self.zero_mean:
            eigs[0] = 0.0  # remove constant mode → zero-mean process
        return eigs

    def _build_map(self):
        """Precompute Phi where field = xi @ Phi.T.

        Phi[n, k] = sqrt(d/2) * M_idct[n, k] * sqrt(lambda_k)

        M_idct is the ortho-normalised IDCT-2 matrix:
            M[n, 0] = 1/sqrt(d)
            M[n, k] = sqrt(2/d) * cos(pi * k * (2n + 1) / (2d))  for k > 0
        """
        d = self.dim_true
        n = torch.arange(d, dtype=torch.float64).unsqueeze(1)   # (d, 1)
        k = torch.arange(d, dtype=torch.float64).unsqueeze(0)   # (1, d)
        M = torch.sqrt(torch.tensor(2.0 / d)) * torch.cos(
            torch.pi * k * (2 * n + 1) / (2 * d)
        )                                                         # (d, d)
        M[:, 0] = 1.0 / torch.sqrt(torch.tensor(float(d)))      # k=0 correction
        sqrt_eigs = torch.sqrt(self.eigenvalues())               # (d,)
        return torch.sqrt(torch.tensor(d / 2.0)) * M * sqrt_eigs.unsqueeze(0)  # (d, d)

    # ------------------------------------------------------------------
    # Coeff-to-field mapping
    # ------------------------------------------------------------------

    def coeff_to_field_map(self):
        """Return Phi (dim_true x dim_true): field = xi @ Phi.T."""
        return self._Phi

    def coefficients_to_field(self, xi):
        """Evaluate KL expansion on the full grid (no noise).

        Parameters
        ----------
        xi : (N, dim_true) tensor of standard-normal draws

        Returns
        -------
        field : (N, dim_true) noiseless field values
        """
        return xi.to(self._Phi) @ self._Phi.T

    # ------------------------------------------------------------------
    # Sampling  (noise-free — noise belongs in the modality)
    # ------------------------------------------------------------------

    def sample_coefficients(self, N):
        """Draw N raw epsilon vectors ~ N(0, I). Returns (N, dim_true)."""
        return torch.randn(N, self.dim_true, dtype=torch.float64)

    def sample_field(self, N):
        """Draw N noiseless field samples on the full grid. Returns (N, dim_true)."""
        return self.coefficients_to_field(self.sample_coefficients(N))

    # ------------------------------------------------------------------
    # Moments (clean — no observation noise)
    # ------------------------------------------------------------------

    def mean_field(self):
        """Prior mean of the field on the full grid."""
        if not self.zero_mean:
            scale = float(self.tau ** 2) ** (-self.alpha / 2)
            return scale * torch.ones(self.dim_true, dtype=torch.float64)
        return torch.zeros(self.dim_true, dtype=torch.float64)

    def cov_field_clean(self):
        """Clean prior covariance on full grid: Phi @ Phi.T. Shape (dim_true, dim_true)."""
        return self._Phi @ self._Phi.T

    def mean_coeff(self):
        """Prior mean of xi: zeros. Shape (dim_true,)."""
        return torch.zeros(self.dim_true, dtype=torch.float64)

    def cov_coeff(self):
        """Prior covariance of xi: identity. Shape (dim_true, dim_true)."""
        return torch.eye(self.dim_true, dtype=torch.float64)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gen = GaussianProcess1D(alpha=2, tau=3, dim_true=200)
    x = gen.spatial_domain().numpy()
    fields = gen.sample_field(3).numpy()

    for f in fields:
        plt.plot(x, f)
    plt.title("GaussianProcess1D — sample fields")
    plt.tight_layout()
    plt.show()
