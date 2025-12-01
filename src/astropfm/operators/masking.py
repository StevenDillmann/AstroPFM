"""
masking.py
____________________________________________________________________________________________________

Description: Masking operators for data validity and training splits.
Author: Steven Dillmann
Date: 2025-11-30
"""

# === Setup ========================================================================================

import jax
import jax.numpy as jnp

# === Main =========================================================================================


class MaskOperator:
    """
    Base class for applying binary masks to data.
    """

    def __init__(self, mask: jnp.ndarray):
        """
        Args:
            mask: Boolean or binary array where 1/True indicates valid data.
        """
        self.mask = mask.astype(bool)

    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        """
        Apply mask to image (sets invalid pixels to 0).
        """
        return jnp.where(self.mask, image, 0.0)


class RandomHoldoutMask(MaskOperator):
    """
    Dynamically creates a training mask by holding out a random fraction of pixels.
    Useful for Posterior Predictive Checks (PPC) and self-supervised validation.
    """

    def __init__(self, base_mask: jnp.ndarray, holdout_fraction: float = 0.2, seed: int = 42):
        """
        Args:
            base_mask: The original validity mask (e.g., detector readout).
            holdout_fraction: Fraction of *valid* pixels to hide (0.0 to 1.0).
            seed: Random seed for reproducibility.
        """
        self.base_mask = base_mask.astype(bool)
        self.holdout_fraction = holdout_fraction
        self.key = jax.random.PRNGKey(seed)

        # Generate the specific holdout mask immediately
        self.training_mask = self._generate_mask()
        super().__init__(self.training_mask)

    def _generate_mask(self) -> jnp.ndarray:
        """Generate the training mask (subset of base_mask)."""
        # Create a random map of same shape
        random_vals = jax.random.uniform(self.key, shape=self.base_mask.shape)

        # Keep pixel if it is valid AND random_val > holdout_fraction
        # (i.e., we DROP pixels where random_val <= holdout_fraction)
        keep_mask = random_vals > self.holdout_fraction

        return self.base_mask & keep_mask
