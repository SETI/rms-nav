=====================
Navigation Techniques
=====================

A navigation technique consumes a subset of the per-image feature set plus the per-image
navigation context and produces one technique result carrying a calibrated translation
offset, a covariance, a confidence, and a typed diagnostics record.  The orchestrator's
ensemble reconciler combines every technique's result into a single navigation result.

Concrete techniques group by the feature family they exploit: body techniques (limb,
terminator, disc, blob), ring techniques (edge, annulus), star techniques (field, unique
match, refine), and the interactive manual technique.  Several techniques share common
infrastructure for distance-transform fitting, image-side derivatives, confidence
calibration, feasibility reporting, and per-technique diagnostics; those topics each have
their own page.

.. toctree::
   :maxdepth: 1
   :caption: Techniques

   dev_guide_techniques_body_limb
   dev_guide_techniques_body_terminator
   dev_guide_techniques_body_disc
   dev_guide_techniques_body_blob
   dev_guide_techniques_ring_edge
   dev_guide_techniques_ring_annulus
   dev_guide_techniques_star_field
   dev_guide_techniques_star_unique_match
   dev_guide_techniques_star_refine
   dev_guide_techniques_manual

.. toctree::
   :maxdepth: 1
   :caption: Shared Infrastructure

   dev_guide_techniques_dt_fitting
   dev_guide_techniques_image_derivatives
   dev_guide_techniques_confidence
   dev_guide_techniques_feasibility
   dev_guide_techniques_diagnostics
