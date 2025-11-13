# Simulated Body Generator GUI

This directory contains a Qt6-based GUI application for experimenting with the `create_simulated_body` function from `nav.support.sim`.

## Requirements

- PyQt6
- numpy
- PIL (Pillow)

To install PyQt6:
```bash
pip install PyQt6
```

## Usage

Run the GUI application:
```bash
python sim_body_gui.py
```

Or make it executable and run directly:
```bash
chmod +x sim_body_gui.py
./sim_body_gui.py
```

## Features

- **Real-time parameter adjustment** with debounced updates
- **Visual aids** (toggleable):
  - Center point (red circle)
  - Semi-major axis (green line)
  - Semi-minor axis (blue line)
  - Illumination direction arrow (yellow)
- **Save/Load parameters** as JSON files
- **Export image** as PNG
- **All angles displayed in degrees** (converted to radians internally)

## Parameter Controls

- **Image Size**: Dimensions in pixels (V = height, U = width)
- **Ellipse Geometry**: Semi-major/minor axes, center position, rotation
- **Illumination**: Illumination angle (0-360°) and phase angle (0-180°)
- **Surface Features**: Edge roughness (mean, std dev) and internal crater density
- **Quality**: Anti-aliasing level (0-1)

## File Formats

- **Parameters (JSON)**: Saves all parameter values in degrees for angles
- **Image (PNG)**: Grayscale image with values 0-255
