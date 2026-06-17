================
Simulated Images
================

RMS-NAV supports simulated images created with the
``nav_create_simulated_image`` GUI. Simulated images share the same
navigation pipeline as real images; they are selected by passing the
``sim`` dataset name and a path to a JSON parameter file on the command
line.

This page summarizes how to run navigation with a simulated image and
points at the full reference.

Creating a simulated image
==========================

Launch the interactive GUI with:

.. code-block:: bash

   nav_create_simulated_image

The GUI lets you:

* Set image dimensions and offset parameters.
* Configure background noise and random background stars.
* Add individual stars with specific properties.
* Add planetary bodies with customizable shapes, rotations, and surface
  features.
* Add planetary rings with elliptical edges.
* Preview the simulated image in real-time.
* Save the configuration as a JSON parameter file.

Running navigation on a simulated image
=======================================

To run navigation on a saved JSON file, use the ``sim`` dataset name and
supply the path to the JSON parameter file:

.. code-block:: bash

   nav_offset sim /path/to/simulated_image.json

You can also select specific navigation models and techniques:

.. code-block:: bash

   nav_offset sim /path/to/simulated_image.json \
     --nav-models stars,rings \
     --nav-techniques correlate_all

JSON parameter file reference
=============================

The full description of the JSON file structure -- every supported
top-level field, star parameter, body parameter, ring parameter, and ring
edge mode -- is documented once under
:doc:`/introduction_simulated_images`.
