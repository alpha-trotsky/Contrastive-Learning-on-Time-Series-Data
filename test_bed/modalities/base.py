# modalities/base.py

from abc import ABC, abstractmethod


class PairSampler(ABC):
    """Abstract interface for paired-modality data sources.

    A PairSampler produces matched (u, v) pairs that share an underlying
    latent variable.  Concrete subclasses encode a specific observation
    model (e.g. field observations paired with spectral coefficients).

    Subclasses must implement:
      - sample_pair(N)  : draw N paired samples
      - dim_u           : dimensionality of u
      - dim_v           : dimensionality of v

    Optional analytical methods (implement when closed-form is available,
    raise NotImplementedError otherwise).  The theory layer consumes these
    to compute predicted optimal encoders.
    """

    @abstractmethod
    def sample_pair(self, N):
        """Draw N independent (u, v) pairs from the joint distribution.

        Returns
        -------
        u : (N, dim_u) tensor
        v : (N, dim_v) tensor
        """
        ...

    @property
    @abstractmethod
    def dim_u(self):
        """Dimensionality of the u modality."""
        ...

    @property
    @abstractmethod
    def dim_v(self):
        """Dimensionality of the v modality."""
        ...

    # ------------------------------------------------------------------
    # Optional analytical triple — subclasses override when tractable
    # ------------------------------------------------------------------

    def C_uu(self):
        """Marginal covariance of u.  Shape (dim_u, dim_u)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a closed-form C_uu."
        )

    def C_vv(self):
        """Marginal covariance of v.  Shape (dim_v, dim_v)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a closed-form C_vv."
        )

    def C_uv(self):
        """Cross-covariance E[u v^T].  Shape (dim_u, dim_v)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a closed-form C_uv."
        )
