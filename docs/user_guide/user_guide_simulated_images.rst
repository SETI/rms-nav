================
Simulated Images
================

RMS-NAV includes an image simulator that synthesizes spacecraft frames -- stars,
planetary bodies, and rings, with a realistic detector model -- for arbitrary
geometry. It exists to **test and validate the navigation pipeline**: because a
simulated frame's true pointing offset is known by construction, it is the only
kind of frame whose navigation answer can be checked exactly. The simulator backs
the algorithmic-invariant tests, the regression baselines, and the
single-variable sensitivity sweeps that verify the navigator behaves as expected.

As a user navigating real spacecraft images you do not need the simulator; the
results you rely on are validated with it on your behalf. If you are extending or
calibrating the pipeline and want to use the simulator directly -- the scene
formats, the GUI, and how to render and navigate synthetic frames -- see the
developer guide's :doc:`/dev_guide/dev_guide_simulator` chapter and the
:doc:`/simulator_report/simulator_report`.
