# RMS-NAV: Spacecraft Image Navigation System

RMS-NAV is a comprehensive navigation system designed for spacecraft imagery processing. It provides tools to analyze images from various space missions (Cassini, Voyager, Galileo, New Horizons) and determine precise positional offsets by comparing observed images with theoretical models of celestial bodies.

## Features

- **Multi-mission support**: Works with Cassini, Voyager, Galileo, and New Horizons imagery
- **Multiple navigation techniques**: Star-based, body-based, and rings-based navigation
- **Automated offset calculation**: Determines precise pointing corrections
- **Visualization tools**: Creates annotated images with identified features
- **Configurable processing**: Customizable parameters for different scenarios

## Installation

### Prerequisites

- Python 3.9 or higher
- SPICE toolkit and kernels for planetary data
- Dependencies listed in `requirements.txt`

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rms-nav.git
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
> ```
> export SETUPTOOLS_ENABLE_FEATURES="legacy-editable"
> ```

## Quick Start

Process a single Cassini image using star navigation:

```bash
python main/nav_main_offset.py COISS --stars-only --image-full-path /path/to/image/N1234567890.IMG
```

Process all Voyager images in a directory:

```bash
python main/nav_main_offset.py VGISS --directory /path/to/voyager/images
```

## Documentation

Comprehensive documentation is available in the `docs` directory:

- [User Guide](docs/user_guide.rst): Complete guide for end users
- [Developer Guide](docs/developer_guide.rst): Information for developers
- [API Reference](docs/api_reference.rst): Detailed API documentation

To build the documentation:

```bash
cd docs
make html
```

The built documentation will be available in `docs/_build/html`.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions to the RMS-NAV project!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Acknowledgments

- This project builds on the OOPS (Object-Oriented Planetary Science) library
- Thanks to NASA/JPL for providing the spacecraft imagery and SPICE kernels
