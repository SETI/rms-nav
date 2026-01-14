================
Simulated Images
================

RMS-NAV supports simulated images created using the simulated image creation GUI
(``nav_create_simulated_image``). These simulated images can include stars,
planetary bodies, and planetary rings for testing navigation algorithms.

Simulated Image Creation GUI
=============================

The simulated image creation GUI provides an interactive interface for designing
and creating simulated images. Launch the GUI with:

.. code-block:: bash

   nav_create_simulated_image

The GUI allows you to:

* Set image dimensions and offset parameters
* Configure background noise and random background stars
* Add individual stars with specific properties
* Add planetary bodies with customizable shapes, rotations, and surface features
* Add planetary rings with elliptical edges
* Preview the simulated image in real-time
* Save the configuration as a JSON parameter file

Simulated Image JSON Format
============================

Simulated images are described using JSON parameter files. The JSON file
contains metadata about the image size, offset, time, and epoch, as well as
arrays describing the simulated features.

Top-Level Fields
----------------

The JSON file has the following top-level structure:

.. code-block:: json

   {
     "size_v": 512,
     "size_u": 512,
     "offset_v": 0.0,
     "offset_u": 0.0,
     "random_seed": 42,
     "background_noise_intensity": 0.0,
     "background_stars_num": 0,
     "background_stars_psf_sigma": 0.9,
     "background_stars_distribution_exponent": 2.5,
     "time": 0.0,
     "epoch": 0.0,
     "closest_planet": null,
     "stars": [],
     "bodies": [],
     "rings": []
   }

Top-Level Field Descriptions
-----------------------------

* ``size_v`` (integer, required): Image height in pixels.

* ``size_u`` (integer, required): Image width in pixels.

* ``offset_v`` (float, default: 0.0): V coordinate offset in pixels to apply
  when rendering the simulated image for navigation.

* ``offset_u`` (float, default: 0.0): U coordinate offset in pixels to apply
  when rendering the simulated image for navigation.

* ``random_seed`` (integer, default: 42): Random seed for reproducible
  generation of background noise, background stars, and body crater patterns.

* ``background_noise_intensity`` (float, default: 0.0): Standard deviation of
  Gaussian background noise (0.0 to 1.0). Higher values add more noise to the
  image.

* ``background_stars_num`` (integer, default: 0): Number of random background
  stars to add (0 to 1000). Stars are randomly positioned and have random
  intensities following a power law distribution.

* ``background_stars_psf_sigma`` (float, default: 0.9): Point spread function
  (PSF) sigma value for background stars in pixels.

* ``background_stars_distribution_exponent`` (float, default: 2.5): Power law
  exponent for background star intensity distribution. Higher values make
  dimmer stars more common.

* ``time`` (float, default: 0.0): Current time in TDB seconds for ring edge
  calculations.

* ``epoch`` (float, default: 0.0): Epoch time in TDB seconds for ring edge
  calculations.

* ``closest_planet`` (string or null, optional): Name of the closest planet
  (e.g., "SATURN") for ring model selection.

* ``stars`` (array, default: []): Array of star parameter dictionaries (see
  below).

* ``bodies`` (array, default: []): Array of body parameter dictionaries (see
  below).

* ``rings`` (array, default: []): Array of ring parameter dictionaries (see
  below).

Star Parameters
---------------

Each star in the ``stars`` array is a dictionary with the following fields:

* ``name`` (string, required): Unique identifier for the star.

* ``v`` (float, required): V coordinate of the star center in pixels.

* ``u`` (float, required): U coordinate of the star center in pixels.

* ``vmag`` (float, default: 3.0): Visual magnitude of the star.

* ``spectral_class`` (string, default: "G2"): Spectral class of the star (e.g.,
  "G2", "K5", "M0").

* ``psf_sigma`` (float, default: 1.0): Point spread function sigma value in
  pixels.

* ``psf_size`` (array of 2 integers, default: [11, 11]): PSF size in pixels as
  [height, width].

Body Parameters
---------------

Each body in the ``bodies`` array is a dictionary with the following fields:

* ``name`` (string, required): Unique identifier for the body.

* ``center_v`` (float, required): V coordinate of the body center in pixels.

* ``center_u`` (float, required): U coordinate of the body center in pixels.

* ``axis1`` (float, default: 100.0): First axis (semi-major axis) in pixels.

* ``axis2`` (float, default: 80.0): Second axis (semi-minor axis) in pixels.

* ``axis3`` (float): Third axis (depth) in pixels. GUI default: 80.0; if not
  provided programmatically, defaults to min(axis1, axis2).

* ``rotation_z`` (float, default: 0.0): Rotation around Z axis in degrees.

* ``rotation_tilt`` (float, default: 0.0): Tilt rotation in degrees.

* ``illumination_angle`` (float, default: 0.0): Illumination angle in degrees.

* ``phase_angle`` (float, default: 0.0): Phase angle in degrees.

* ``crater_fill`` (float, default: 0.0): Crater fill factor (0.0 to 1.0). Higher
  values add more craters to the surface.

* ``crater_min_radius`` (float, default: 0.05): Minimum crater radius as a
  fraction of body size.

* ``crater_max_radius`` (float, default: 0.25): Maximum crater radius as a
  fraction of body size.

* ``crater_power_law_exponent`` (float, default: 3.0): Power law exponent for
  crater size distribution.

* ``crater_relief_scale`` (float, default: 0.6): Scale factor for crater depth
  relief.

* ``anti_aliasing`` (float, default: 0.5): Anti-aliasing factor for body edge
  smoothing.

* ``range`` (float, required): Range value for depth ordering. Bodies with
  smaller range values appear in front of bodies with larger range values.

* ``seed`` (integer, optional): Random seed for this body's crater generation.
  If not specified, uses the top-level ``random_seed``.

Ring Parameters
---------------

Each ring in the ``rings`` array is a dictionary with the following fields:

* ``name`` (string, required): Unique identifier for the ring.

* ``feature_type`` (string, required): Either ``"RINGLET"`` (bright ring) or
  ``"GAP"`` (dark gap).

* ``center_v`` (float, required): V coordinate of the ring center in pixels.

* ``center_u`` (float, required): U coordinate of the ring center in pixels.

* ``inner_data`` (array, optional): List of mode dictionaries for the inner
  edge. Must include at least a mode 1 dictionary (see below).

* ``outer_data`` (array, optional): List of mode dictionaries for the outer
  edge. Must include at least a mode 1 dictionary (see below).

* ``shading_distance`` (float, default: 20.0): Distance in pixels for edge
  fading.

* ``range`` (float, required): Range value for depth ordering. Rings with
  smaller range values appear in front of rings with larger range values.

Ring Edge Mode Data
-------------------

Each edge's mode data (``inner_data`` and ``outer_data``) must include at least
a mode 1 dictionary with the following fields:

* ``mode`` (integer, required): Must be 1 for the base mode.

* ``a`` (float, required): Semi-major axis in pixels.

* ``rms`` (float, optional): Root mean square value (not currently used in
  rendering).

* ``ae`` (float, default: 0.0): Eccentricity times semi-major axis in pixels.

* ``long_peri`` (float, default: 0.0): Longitude of pericenter in degrees.

* ``rate_peri`` (float, default: 0.0): Rate of precession in degrees/day.

Additional modes (mode 2 and higher) can be added for more complex edge shapes,
but mode 1 is required and defines the base elliptical edge.

Running Navigation with Simulated Images
=========================================

To run navigation on a simulated image, use the ``sim`` dataset name and
provide the path to the JSON parameter file:

.. code-block:: bash

   nav_offset sim /path/to/simulated_image.json

You can also specify navigation models and techniques:

.. code-block:: bash

   nav_offset sim /path/to/simulated_image.json \
     --nav-models stars,rings \
     --nav-techniques correlate_all

Ring Navigation Models
----------------------

When processing simulated images, RMS-NAV automatically detects rings defined
in the JSON parameters and uses the ``NavModelRingsSimulated`` class to create
navigation models. This class:

* Reads ring parameters directly from the JSON file (via the observation's
  ``sim_rings`` attribute)
* Renders rings using the same anti-aliasing and edge fading algorithms as real
  ring models
* Creates annotations for ring edges using the same annotation system as real
  rings
* Shares common functionality with ``NavModelRings`` through the
  ``NavModelRingsBase`` base class

The ring model system uses a base class architecture:

* ``NavModelRingsBase``: Abstract base class providing shared functionality for
  anti-aliasing, edge fading, and annotation creation
* ``NavModelRings``: Subclass for real rings computed from ephemeris data and
  YAML configuration files
* ``NavModelRingsSimulated``: Subclass for simulated rings defined in the GUI
  and JSON parameter files

Both ring model types produce consistent navigation models and annotations,
ensuring that simulated images can be used to test navigation algorithms with
the same quality as real observations.

Example JSON File
=================

Here is a complete example of a simulated image JSON file:

.. code-block:: json

   {
     "size_v": 512,
     "size_u": 512,
     "offset_v": 2.5,
     "offset_u": -1.3,
     "random_seed": 42,
     "background_noise_intensity": 0.01,
     "background_stars_num": 50,
     "background_stars_psf_sigma": 0.9,
     "background_stars_distribution_exponent": 2.5,
     "time": 0.0,
     "epoch": 0.0,
     "closest_planet": "SATURN",
     "stars": [
       {
         "name": "Star1",
         "v": 256.0,
         "u": 256.0,
         "vmag": 3.0,
         "spectral_class": "G2",
         "psf_sigma": 1.0,
         "psf_size": [11, 11]
       }
     ],
     "bodies": [
       {
         "name": "Saturn",
         "center_v": 256.0,
         "center_u": 256.0,
         "axis1": 200.0,
         "axis2": 180.0,
         "axis3": 180.0,
         "rotation_z": 0.0,
         "rotation_tilt": 0.0,
         "illumination_angle": 0.0,
         "phase_angle": 0.0,
         "crater_fill": 0.0,
         "crater_min_radius": 0.05,
         "crater_max_radius": 0.25,
         "crater_power_law_exponent": 3.0,
         "crater_relief_scale": 0.6,
         "anti_aliasing": 0.5,
         "range": 1.0
       }
     ],
     "rings": [
       {
         "name": "Ring1",
         "feature_type": "RINGLET",
         "center_v": 256.0,
         "center_u": 256.0,
         "inner_data": [
           {
             "mode": 1,
             "a": 250.0,
             "rms": 1.0,
             "ae": 0.0,
             "long_peri": 0.0,
             "rate_peri": 0.0
           }
         ],
         "outer_data": [
           {
             "mode": 1,
             "a": 300.0,
             "rms": 1.0,
             "ae": 0.0,
             "long_peri": 0.0,
             "rate_peri": 0.0
           }
         ],
         "shading_distance": 20.0,
         "range": 1000.0
       }
     ]
   }
