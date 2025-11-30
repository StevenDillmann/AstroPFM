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

    def __init__(self, kernel: jnp.ndarray, normalize: bool = True):
        if normalize:
            self.kernel = kernel / jnp.maximum(jnp.sum(kernel), 1e-12)
        else:
            self.kernel = kernel

    def __call__(self, signal: jnp.ndarray, mode: str = "same", axes: tuple = None) -> jnp.ndarray:
        return self.convolve(signal, mode=mode, axes=axes)

    def convolve(self, signal: jnp.ndarray, mode: str = "same", axes: tuple = None) -> jnp.ndarray:
        return fftconvolve(signal, self.kernel, mode=mode, axes=axes)
