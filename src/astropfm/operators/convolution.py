"""
convolution.py
____________________________________________________________________________________________________

Description: Convolution operators for AstroPFM.
Author: Steven Dillmann
Date: 2025-11-30
"""

# === Setup ========================================================================================

import jax.numpy as jnp
from jax.scipy.signal import fftconvolve

# === Main =========================================================================================


class Convolver:
    """
    Convolver class. This class is used to convolve an image with a PSF.
    """

    def __init__(self, kernel: jnp.ndarray, normalize: bool = True, mode: str = "same", axes: tuple | None = None):
        self.kernel = kernel
        self.normalize = normalize
        self.mode = mode
        self.axes = axes

        if self.normalize:
            self.kernel = self.kernel / jnp.maximum(jnp.sum(self.kernel), 1e-12)

    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        return self.convolve(s)

    def convolve(self, s: jnp.ndarray) -> jnp.ndarray:
        return fftconvolve(s, self.kernel, mode=self.mode, axes=self.axes)
