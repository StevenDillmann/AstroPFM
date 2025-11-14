# === Setup ========================================================================================

import jax
import jax.numpy as jnp
import nifty8.re as jft

# === Main =========================================================================================


class BaseGP(jft.Model):
    """A collection of independent correlated field models for each channel"""

    def __init__(
        self,
        n_channels,
        shape,
        distances,
        offset_mean,
        offset_std,
        fluctuations,
        loglogavgslope,
        flexibility,
        asperity=None,
        name="cf",
        prefix="",
    ):
        correlated_fields = get_correlated_field(
            shape,
            distances,
            offset_mean,
            offset_std,
            fluctuations,
            loglogavgslope,
            flexibility,
            asperity,
            name=name,
            prefix=prefix,
        )
        domain = {
            k: jft.ShapeWithDtype((n_channels,) + correlated_fields.domain[k].shape, jnp.float64)
            for k in correlated_fields.domain.keys()
        }
        self._correlated_fields = correlated_fields
        self._n_channels = n_channels
        super().__init__(domain=domain, white_init=True)

    @property
    def n_channels(self):
        return self._n_channels

    def __call__(self, x):
        return jax.vmap(self.correlated_fields)(x)


class MixtureGP(jft.Model):
    """A mixture of Gaussian processes with a global mixing matrix"""

    def __init__(
        self,
        base_models,
        mix_mode="cholesky",
        mix_diag=(0.0, 1.0),
        mix_off_diag=(0.0, 1.0),
        mix_full=(0.0, 1.0),
        mix_offset=(0.0, 1.0),
    ):
        self.base_models = base_models
        self.mix_mode = mix_mode
        domain = self.base_models.domain
        init = self.base_models.init
        n_channels = self.base_models.n_channels

        if self.mix_mode == "full":
            domain, init = self._build_full_mixing_matrix(n_channels, domain, init, mix_full)
        elif self.mix_mode == "diag":
            domain, init = self._build_diag_mixing_matrix(n_channels, domain, init, mix_diag, mix_off_diag)

        domain, init = self._build_mixing_offset(n_channels, domain, init, mix_offset)
        super().__init__(domain=domain, init=init)

    def _build_full_mixing_matrix(self, n_channels, domain, init, mix_full):
        self.mixing_matrix = jft.NormalPrior(
            mix_full[0],
            mix_full[1],
            shape=(n_channels, n_channels),
            name="mixture_matrix",
        )
        return domain | self.mixing_matrix.domain, init | self.mixing_matrix.init

    def _build_diag_mixing_matrix(self, n_channels, domain, init, mix_diag, mix_off_diag):
        self.mixing_diag = jft.NormalPrior(
            mix_diag[0],
            mix_diag[1],
            shape=(n_channels,),
            name="mixing_diag",
        )
        self.mixing_off_diag = jft.NormalPrior(
            mix_off_diag[0],
            mix_off_diag[1],
            shape=(n_triangular_lower(n_channels),),
            name="mixing_off_diag",
        )
        return (
            domain | self.mixing_diag.domain | self.mixing_off_diag.domain,
            init | self.mixing_diag.init | self.mixing_off_diag.init,
        )

    def _build_mixing_offset(self, n_channels, domain, init, mix_offset):
        self.mixing_offset = jft.NormalPrior(
            mix_offset[0],
            mix_offset[1],
            shape=(n_channels,),
            name="mixing_offset",
        )
        return domain | self.mixing_offset.domain, init | self.mixing_offset.init

    def __call__(self, state):
        base_outputs = self.base_models({k: state[k] for k in self.base_models.domain.keys()})

        if self.mix_mode == "full":
            mixing_matrix = self.mixing_matrix(state)
        elif self.mix_mode == "diag":
            diag = self.mixing_diag(state)
            off_diag = self.mixing_off_diag(state)
            mixing_matrix = assemble_cholesky_matrix(diag, off_diag)

        mixture_output = jnp.tensordot(mixing_matrix, base_outputs, axes=(1, 0))
        return jnp.exp(mixture_output + self.mixing_offset(state)[:, None, None])


# === Utilities ====================================================================================


def get_correlated_field(
    shape,
    distances,
    offset_mean,
    offset_std,
    fluctuations,
    loglogavgslope,
    flexibility,
    asperity,
    name="cf",
    prefix="",
):
    """Create an instance of a correlated field model"""
    cf_zm = dict(offset_mean=offset_mean, offset_std=offset_std)
    cf_fl = dict(
        fluctuations=fluctuations,
        loglogavgslope=loglogavgslope,
        flexibility=flexibility,
        asperity=asperity,
    )
    cfm = jft.CorrelatedFieldMaker(name)
    cfm.set_amplitude_total_offset(**cf_zm)
    cfm.add_fluctuations(shape, distances=distances, **cf_fl, prefix=prefix, non_parametric_kind="amplitude")
    correlated_field = cfm.finalize()
    return correlated_field


def n_triangular_lower(n):
    """Number of elements in the lower triangular part of a square matrix"""
    return (n * (n - 1)) // 2


def assemble_cholesky_matrix(diag, off_diag):
    """Assemble a Cholesky matrix from a diagonal and off-diagonal vector"""
    n_channels = len(diag)
    L = jnp.zeros((n_channels, n_channels))
    L = L.at[jnp.diag_indices(n_channels)].set(1.0)
    tril_indices = jnp.tril_indices(n_channels, k=-1)
    L = L.at[tril_indices].set(off_diag)
    L = L * (jnp.exp(diag))[:, None]
    return L
