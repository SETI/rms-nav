============
Orchestrator
============

The orchestrator is the top-level driver that turns one observation into one navigation
result.  It is a small subsystem of cooperating components rather than a single class: a
driver that runs the two-pass pipeline, a per-image state container, a final-output
container, an ensemble reconciler, a quick-fail image classifier, a per-instrument
settings resolver, a reproducibility envelope, a JSON curator, and a per-feature
post-mortem entry.

Each public component has its own page below.

.. toctree::
   :maxdepth: 1

   dev_guide_orchestrator_orchestrator
   dev_guide_orchestrator_nav_context
   dev_guide_orchestrator_nav_result
   dev_guide_orchestrator_ensemble
   dev_guide_orchestrator_image_classifier
   dev_guide_orchestrator_instrument_config
   dev_guide_orchestrator_provenance
   dev_guide_orchestrator_curator
   dev_guide_orchestrator_feature_summary
