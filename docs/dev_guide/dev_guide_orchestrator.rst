==========================================================
Orchestrator Subsystem
==========================================================

The orchestrator subsystem (:mod:`nav.nav_orchestrator`) is the top-level driver that
turns one observation into one
:class:`~nav.nav_orchestrator.nav_result.NavResult`. The driver class
:class:`~nav.nav_orchestrator.orchestrator.NavOrchestrator` runs every registered
:class:`~nav.nav_model.nav_model.NavModel`'s feature extraction, applies the per-feature
reliability gate, runs every feasible
:class:`~nav.nav_technique.nav_technique.NavTechnique` in two passes (prior-free, then
prior-required), reconciles per-technique results into a final answer via the
:func:`~nav.nav_orchestrator.ensemble.ensemble` combine, and emits a single
:class:`~nav.nav_orchestrator.nav_result.NavResult` carrying the headline offset plus the
full diagnostic envelope.

The subsystem is more than a single class: it is a small package of cooperating
dataclasses, helpers, and a pipeline. Each public component has its own page below.

.. toctree::
   :maxdepth: 4
   :caption: Driver

   dev_guide_orchestrator_orchestrator

.. toctree::
   :maxdepth: 4
   :caption: Per-image dataclasses

   dev_guide_orchestrator_nav_context
   dev_guide_orchestrator_nav_result
   dev_guide_orchestrator_feature_summary
   dev_guide_orchestrator_provenance

.. toctree::
   :maxdepth: 4
   :caption: Per-image helpers

   dev_guide_orchestrator_image_classifier
   dev_guide_orchestrator_instrument_config
   dev_guide_orchestrator_ensemble
   dev_guide_orchestrator_curator

The :mod:`nav.nav_orchestrator.image_derivatives` module also lives in this
package; the orchestrator runs it once per image to populate the shared gradient and
edge-distance-transform products on
:class:`~nav.nav_orchestrator.nav_context.NavContext`. Because the module exists to feed
the distance-transform techniques, its dedicated page is filed under the techniques
chapter at :doc:`dev_guide_techniques_image_derivatives`.
