======================
Body Navigation Models
======================

The body navigation models render the predicted appearance of a planetary body and emit
the limb, terminator, disc, and blob features that body techniques consume.  Two concrete
models make up the family: the catalog-driven model, which renders the body from SPICE
prediction and per-body shape parameters, and the simulated model, which renders the body
from operator-supplied parameters for test-image generation.

- :class:`~nav.nav_model.nav_model_body.NavModelBody` — catalog-driven body navigation;
  one instance per body whose bounding box overlaps the extended field of view.
- :class:`~nav.nav_model.nav_model_body_simulated.NavModelBodySimulated` — simulated body
  model rendered from operator-supplied parameters.

.. toctree::
   :maxdepth: 1

   dev_guide_navigation_models_body
   dev_guide_navigation_models_body_simulated
