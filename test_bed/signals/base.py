# signals/base.py

from abc import ABC, abstractmethod
import torch


class SignalGenerator(ABC):
    """Abstract base class for signal generators.
    
    A signal generator produces realizations of an underlying random
    object — typically a function, field, or trajectory — along with
    the deterministic machinery needed to evaluate or analyze it.
    
    Concrete subclasses implement specific signal families (Gaussian
    processes, dynamical systems, parametric waveforms, etc.). Modalities
    consume a signal generator to construct paired observations (u, v).
    
    Subclasses must implement:
      - `sample_coefficients(N)`: draw N samples of the underlying randomness
      - `coefficients_to_field(xi)`: deterministic map from randomness to
        a fully-specified realization on the canonical grid
    
    Optional analytical methods (implement when available, raise
    NotImplementedError otherwise):
      - `coeff_to_field_map()`: explicit linear operator Phi if linear
      - `cov_field_clean()`: prior covariance of the realization
      - `mean_field()`: prior mean of the realization
    
    These optional methods enable closed-form computation of modality
    statistics, which the theory module uses to predict optimal encoder
    behavior under each loss variant.
    """
    
    @property
    @abstractmethod
    def dim_true(self):
        """Dimensionality of the realization on its canonical grid."""
        ...
    
    @abstractmethod
    def sample_coefficients(self, N):
        """Draw N independent realizations of the underlying randomness.
        
        Returns
        -------
        xi : (N, ?) tensor — the raw stochastic input
        """
        ...
    
    @abstractmethod
    def coefficients_to_field(self, xi):
        """Map a batch of stochastic inputs to realizations on the grid.
        
        Deterministic — all randomness lives in xi.
        
        Parameters
        ----------
        xi : (N, ?) tensor
        
        Returns
        -------
        field : (N, dim_true) tensor — realization on the canonical grid
        """
        ...
    
    # Optional analytical interface — concrete classes implement when
    # the math is tractable.
    
    def coeff_to_field_map(self):
        """Linear operator Phi such that field = xi @ Phi.T.
        Override when the coefficient-to-field map is linear."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a linear "
            f"coefficient-to-field map."
        )
    
    def cov_field_clean(self):
        """Prior covariance of the realization (no observation noise).
        Override when analytically available."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a closed-form "
            f"prior covariance."
        )
    
    def mean_field(self):
        """Prior mean of the realization on the canonical grid."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a closed-form "
            f"prior mean."
        )