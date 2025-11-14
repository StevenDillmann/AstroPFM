# AstroPFM: Probabilistic Foundation Model for Astronomy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Future badges to add when ready:
[![PyPI version](https://badge.fury.io/py/astropfm.svg)](https://badge.fury.io/py/astropfm)
[![Documentation Status](https://readthedocs.org/projects/astropfm/badge/?version=latest)](https://astropfm.readthedocs.io/en/latest/?badge=latest)
[![arXiv](https://img.shields.io/badge/arXiv-2XXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2XXX.XXXXX)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Coverage](https://codecov.io/gh/StevenDillmann/AstroPFM/branch/main/graph/badge.svg)](https://codecov.io/gh/StevenDillmann/AstroPFM)
-->

A unified probabilistic foundation model that integrates astronomical observations across diverse wavelengths and spatial scales using multi-output Gaussian Processes and Information Field Theory.

## Overview

Unlike traditional approaches that analyze individual surveys separately, or recent deterministic foundation models like AION, AstroPFM provides:

- **Full posterior distributions** of flux density across wavelength and spatial location
- **Explicit modeling** of instrumental effects (PSF, noise, calibration)
- **Rigorous uncertainty quantification** for Bayesian inference
- **Multi-wavelength integration** from X-ray to radio observations

Built on [NIFTy](https://ift.pages.mpcdf.de/nifty/) (Numerical Information Field Theory), AstroPFM enables probabilistic inference on spatially-correlated astrophysical fields.

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/AstroPFM.git
cd AstroPFM

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# Install in development mode
uv pip install -e ".[dev]"
```

## Quick Start
```python
import astropfm

# Coming soon...
```

## Development Status

🚧 **Early Development** - This project is in active development. APIs may change.

## Citation

If you use AstroPFM in your research, please cite:
```bibtex
@article{dillmann2024astropfm,
  title={AstroPFM: A Unified Probabilistic Foundation Model for Astronomy across Wavelength and Scale},
  author={Dillmann, Steven and Baron, Dalya and Frank, Philipp and Clark, Susan and Wechsler, Risa},
  year={2024},
  institution={Stanford University}
}
```

## Team

- **Steven Dillmann** - Stanford AI Lab, KIPAC
- **Dalya Baron** - KIPAC, Stanford Data Science
- **Philipp Frank** - KIPAC, Stanford Data Science
- **Susan Clark** - KIPAC, Stanford Data Science
- **Risa Wechsler** - KIPAC, Stanford Data Science

## License

MIT License - see [LICENSE](LICENSE) file for details.
