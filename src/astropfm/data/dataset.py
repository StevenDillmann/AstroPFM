"""
dataset.py
____________________________________________________________________________________________________

Description: Dataset class for AstroPFM.
Author: Steven Dillmann
Date: 2025-11-20
"""

# === Setup ========================================================================================

import glob
import os
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

# === Main =========================================================================================


@dataclass
class Dataset:
    """A multi-channel dataset for AstroPFM.

    Attributes:
        data: A stack of images.
        wcs: A list of WCS objects.
        psfs: A list of PSFs.
        keys: A list of keys.
    """

    data: jnp.ndarray
    wcs: list[WCS]
    psfs: jnp.ndarray
    keys: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization checks and transformations."""
        n_channels = self.data.shape[0]
        if n_channels != len(self.wcs) or n_channels != len(self.psfs) or n_channels != len(self.keys):
            raise ValueError("Number of data, WCS, PSF, and keys must match.")

    @property
    def n_channels(self) -> int:
        """Number of channels in the dataset."""
        return self.data.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the dataset channels."""
        return self.data.shape[1:]

    @property
    def readout(self) -> jnp.ndarray:
        """Readout mask for the dataset."""
        return (jnp.isfinite(self.data) & (self.data > 0)).astype(bool)

    @property
    def distances(self) -> list[tuple[float, float]]:
        """Get pixel scales (dy, dx) in arcsec for each channel."""
        distances = []
        for w in self.wcs:
            scales_deg = proj_plane_pixel_scales(w)
            scales_arcsec = (float(scales_deg[1] * 3600), float(scales_deg[0] * 3600))
            distances.append(scales_arcsec)
        return distances

    def __getitem__(self, key: str | int) -> tuple[jnp.ndarray, WCS, jnp.ndarray]:
        """
        Get item from dataset by channel key (name) or integer index.
        Returns (data_channel, wcs_channel, psf_channel)
        """
        if isinstance(key, str):
            try:
                index = self.keys.index(key)
            except ValueError as err:
                raise KeyError(f"Channel key '{key}' not found in dataset keys.") from err

        elif isinstance(key, int):
            if 0 <= key < self.n_channels:
                index = key
            else:
                raise IndexError(f"Index {key} out of range for {self.n_channels} channels.")
        else:
            raise TypeError(f"Dataset index must be a string (key) or an integer, not {type(key).__name__}.")

        data_channel = self.data[index]
        wcs_channel = self.wcs[index]
        psf_channel = self.psfs[index]

        return data_channel, wcs_channel, psf_channel


def load_dataset(path: str, extension: str = "fits") -> Dataset:
    """Load dataset from directory or file pattern.

    Args:
        Can be a directory path OR a glob pattern.
                         If directory, defaults to searching "*.fits"

    Returns:
        Dataset object with stacked arrays
    """
    # Determine file pattern
    if os.path.isdir(path):
        pattern = os.path.join(path, f"*.{extension}")
    else:
        pattern = path

    # Get files
    files = sorted(glob.glob(pattern))
    keys = [file.split("/")[-1].split(".")[0] for file in files]

    # Load data
    data_list = []
    wcs_list = []
    psf_list = []
    for file in files:
        if file.endswith(".fits"):
            d, w, p = _load_fits(file)
        elif file.endswith((".h5", ".hdf5")):
            d, w, p = _load_hdf5(file)
        else:
            raise ValueError(f"Unsupported file extension: {file}")

        data_list.append(d)
        wcs_list.append(w)
        psf_list.append(p)

    return Dataset(data=jnp.stack(data_list), wcs=wcs_list, psfs=jnp.stack(psf_list), keys=keys)


# === Utilities ====================================================================================


def _load_fits(path: str) -> tuple[np.ndarray, WCS, np.ndarray]:
    """Load data, WCS, and PSF from a single FITS file."""
    with fits.open(path) as hdul:
        # TODO: handle multiple extensions (there is a smarter loader with astropy.io.fits.getdata)
        data = hdul["SCI"].data.astype(np.float64)
        data_hdr = hdul["SCI"].header
        wcs = WCS(data_hdr)
        psf = hdul["PSF"].data.astype(np.float64)
        # TODO: also have WCS for PSF
    return data, wcs, psf


def _load_hdf5(path: str) -> tuple[np.ndarray, WCS, np.ndarray]:
    """Load data, WCS, and PSF from HDF5 file."""
    raise NotImplementedError("HDF5 loading not yet implemented")
