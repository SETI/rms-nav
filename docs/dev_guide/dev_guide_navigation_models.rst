=================
Navigation Models
=================

A navigation model renders the predicted appearance of one scene component and emits the
features that techniques consume.  The abstract base defines three hooks every concrete
model implements: one that populates internal state, one that returns features for
technique consumption, and one that returns annotations for the summary image.  Concrete
models self-register so the orchestrator can discover them and instantiate the right set
for each observation.

The registered families are the body models (catalog-driven and simulated), the ring
models (catalog-driven and simulated), the star model, and the Titan model.  The
body family has its own landing page; the remaining models each have a per-class page.

The API surface is summarised under :doc:`/api_reference/api_nav_model`.

.. toctree::
   :maxdepth: 1

   dev_guide_navigation_models_bodies
   dev_guide_navigation_models_rings
   dev_guide_navigation_models_rings_simulated
   dev_guide_navigation_models_stars
   dev_guide_navigation_models_titan
