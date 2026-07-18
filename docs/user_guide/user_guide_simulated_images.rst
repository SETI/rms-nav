================
Simulated Images
================

SpinDoctor includes an image simulator that synthesizes spacecraft frames -- stars,
planetary bodies, and rings, with a realistic detector model -- for arbitrary
geometry. It exists to **test and validate the navigation pipeline**: because a
simulated frame's true pointing offset is known by construction, it is the only
kind of frame whose navigation answer can be checked exactly. The simulator backs
the algorithmic-invariant tests, the regression baselines, and the
single-variable sensitivity sweeps that verify the navigator behaves as expected.

What such a check measures depends on how the simulated image is built. When the
rendered scene matches the navigator's own model of the sky, recovering the known
offset measures **reproducibility** -- a self-consistency floor -- not accuracy.
The numbers speak to accuracy only to the degree a simulated frame resembles a
real one, which is what each instrument's **realism match** establishes; that
match is what makes sim numbers credible. Sensitivity is reported through
**model-mismatch sweeps**: recovery error measured as a function of how far the
rendered image departs from the navigator's model.

As a user navigating real spacecraft images you do not need the simulator; the
results you rely on are validated with it on your behalf. If you are extending or
calibrating the pipeline and want to use the simulator directly -- the schema v2
scene format, the GUI, and how to render and navigate synthetic frames -- see the
developer guide's :doc:`/dev_guide/dev_guide_simulator` chapter and the
:doc:`/simulator_report/simulator_report`.
