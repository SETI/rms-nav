========
Overview
========

RMS-NAV is a spacecraft image navigation system designed to analyze images from
various space missions and determine precise positional offsets. This overview
provides an introduction to the system architecture, installation, and
command-line tools.

Navigation Pipeline
===================

RMS-NAV follows a three-phase pipeline for processing spacecraft imagery:

1. **Navigation** - Determine pointing offsets by correlating observed images
   with theoretical models of stars, planets, moons, and rings.

2. **Backplanes** - Generate geometric and photometric backplanes (derived
   image products) that provide per-pixel information about the observation
   geometry.

3. **PDS4 Bundle** - Create PDS4-compliant data bundles containing navigation
   results, backplanes, and metadata for archival and distribution.

Each phase builds upon the previous one, with navigation results informing
backplane generation, and both contributing to the final PDS4 bundle.

Installation
============

RMS-NAV can be installed using either ``pip`` or ``pipx``:

Using pip
---------

.. code-block:: bash

   pip install rms-nav

This installs the package and all command-line programs into your Python
environment.

Using pipx
----------

.. code-block:: bash

   pipx install rms-nav

This creates isolated command-line programs that can be run independently of
your Python environment. This is recommended if you want the command-line tools
available system-wide without managing Python dependencies.

Command-Line Programs
=====================

RMS-NAV provides command-line programs that correspond to each phase of the
navigation pipeline:

Navigation Phase
----------------

* ``nav_offset`` - Perform navigation on spacecraft images, determining pointing
  offsets by correlating observed features with theoretical models.

* ``nav_create_simulated_image`` - Create simulated images with stars, bodies,
  and rings for testing navigation algorithms.

Backplanes Phase
----------------

* ``nav_backplanes`` - Generate geometric and photometric backplanes for
  spacecraft images.

* ``nav_backplane_viewer`` - Interactive viewer for examining backplane data.

PDS4 Bundle Phase
-----------------

* ``nav_create_bundle`` - Create PDS4-compliant data bundles containing
  navigation results, backplanes, and metadata. Supports both label generation
  and summary creation.

Cloud Tasks Support
===================

RMS-NAV supports queue-driven processing through cloud tasks for scalable,
distributed processing:

* ``nav_offset_cloud_tasks`` - Cloud tasks worker for navigation processing.

* ``nav_backplanes_cloud_tasks`` - Cloud tasks worker for backplane generation.

* ``nav_create_bundle_cloud_tasks`` - Cloud tasks worker for PDS4 bundle
  creation.

These cloud tasks variants read task payloads from a queue and process batches
of files, making them suitable for large-scale processing in cloud
environments. The standard command-line programs can generate cloud tasks JSON
files using the ``--output-cloud-tasks-file`` option.
