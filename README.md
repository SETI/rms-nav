# RMS-NAV

[![GitHub release; latest by date](https://img.shields.io/github/v/release/SETI/rms-nav)](https://github.com/SETI/rms-nav/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/SETI/rms-nav)](https://github.com/SETI/rms-nav/releases)
[![Test Status](https://img.shields.io/github/actions/workflow/status/SETI/rms-nav/run-tests.yml?branch=main)](https://github.com/SETI/rms-nav/actions)
[![Documentation Status](https://readthedocs.org/projects/rms-nav/badge/?version=latest)](https://rms-nav.readthedocs.io/en/latest/?badge=latest)
[![Code coverage](https://img.shields.io/codecov/c/github/SETI/rms-nav/main?logo=codecov)](https://codecov.io/gh/SETI/rms-nav)
<br />
[![PyPI - Version](https://img.shields.io/pypi/v/rms-nav)](https://pypi.org/project/rms-nav)
[![PyPI - Format](https://img.shields.io/pypi/format/rms-nav)](https://pypi.org/project/rms-nav)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/rms-nav)](https://pypi.org/project/rms-nav)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rms-nav)](https://pypi.org/project/rms-nav)
<br />
[![GitHub commits since latest release](https://img.shields.io/github/commits-since/SETI/rms-nav/latest)](https://github.com/SETI/rms-nav/commits/main/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/SETI/rms-nav)](https://github.com/SETI/rms-nav/commits/main/)
[![GitHub last commit](https://img.shields.io/github/last-commit/SETI/rms-nav)](https://github.com/SETI/rms-nav/commits/main/)
<br />
[![Number of GitHub open issues](https://img.shields.io/github/issues-raw/SETI/rms-nav)](https://github.com/SETI/rms-nav/issues)
[![Number of GitHub closed issues](https://img.shields.io/github/issues-closed-raw/SETI/rms-nav)](https://github.com/SETI/rms-nav/issues)
[![Number of GitHub open pull requests](https://img.shields.io/github/issues-pr-raw/SETI/rms-nav)](https://github.com/SETI/rms-nav/pulls)
[![Number of GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed-raw/SETI/rms-nav)](https://github.com/SETI/rms-nav/pulls)
<br />
![GitHub License](https://img.shields.io/github/license/SETI/rms-nav)
[![Number of GitHub stars](https://img.shields.io/github/stars/SETI/rms-nav)](https://github.com/SETI/rms-nav/stargazers)
![GitHub forks](https://img.shields.io/github/forks/SETI/rms-nav)

RMS-NAV is a comprehensive navigation system designed for spacecraft imagery processing. It provides tools to analyze images from various space missions (Cassini, Voyager, Galileo, New Horizons) and determine precise positional offsets by comparing observed images with theoretical models of celestial bodies.

# Features

- **Multi-mission support**: Works with Cassini, Voyager, Galileo, and New Horizons imagery
- **Multiple navigation techniques**: Star-based, body-based, and rings-based navigation
- **Automated offset calculation**: Determines precise pointing corrections
- **Visualization tools**: Creates annotated images with identified features
- **Configurable processing**: Customizable parameters for different scenarios
- **PDS4 bundle generation**: Creates PDS4-compliant bundles with labels, metadata, and browse products
- **Backplane generation**: Computes per-pixel geometry products (longitude, latitude, angles, etc.)

# Installation

## Prerequisites

- Python 3.10 or higher
- SPICE toolkit and kernels for planetary data
- Dependencies listed in `requirements.txt`

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/SETI/rms-nav.git
   cd rms-nav
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up SPICE kernels:
   - Download the required SPICE kernels for your mission
   - Set the `SPICE_PATH` environment variable to point to your kernels directory:

     ```bash
     export SPICE_PATH=/path/to/your/spice/kernels
     ```

> **Note**: To fix mypy operability with editable pip installs:
> ```bash
> export SETUPTOOLS_ENABLE_FEATURES="legacy-editable"
> ```

# Quick Start

Process a single Cassini image using the installed CLI script:

```bash
nav_offset coiss N1234567890 \
  --pds3-holdings-root /path/to/pds3 \
  --nav-results-root /path/to/nav_results
```

Process all Voyager images within a single PDS3 volume:

```bash
nav_offset vgiss \
  --volumes VGISS_5101 \
  --pds3-holdings-root /path/to/pds3 \
  --nav-results-root /path/to/nav_results
```

Generate backplanes for processed images:

```bash
nav_backplanes coiss_saturn \
  --nav-results-root /path/to/nav_results \
  --backplane-results-root /path/to/backplane_results \
  --volumes COISS_2001
```

Generate PDS4 bundle files:

```bash
nav_create_bundle labels coiss_saturn \
  --nav-results-root /path/to/nav_results \
  --backplane-results-root /path/to/backplane_results \
  --bundle-results-root /path/to/bundle_results \
  --volumes COISS_2001
```

# Documentation

Comprehensive documentation is available in the `docs` directory.

To build the documentation:

```bash
cd docs
make html
```

The built documentation will be available in `docs/_build/html`.

# Contributing

Information on contributing to this package can be found in the
[Contributing Guide](https://github.com/SETI/rms-nav/blob/main/CONTRIBUTING.md).

# Licensing

This code is licensed under the [Apache License v2.0](https://github.com/SETI/rms-nav/blob/main/LICENSE).
