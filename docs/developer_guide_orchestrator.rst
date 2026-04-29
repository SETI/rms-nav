============================================
End-to-End Orchestration and the Summary PNG
============================================

The :class:`~nav.nav_orchestrator.orchestrator.NavOrchestrator` turns one
:class:`~nav.obs.ObsSnapshotInst` into one
:class:`~nav.nav_orchestrator.nav_result.NavResult`.  The driver
:func:`nav.navigate_image_files.navigate_image_files` wraps a single
``ObsClass`` plus an :class:`~nav.dataset.dataset.ImageFiles` batch (size
one) around the orchestrator and adds I/O: a ``_metadata.json`` JSON dump
plus a ``_summary.png`` overlay image.

Pipeline at a glance
====================

.. code-block:: text

    +-------------------------+
    | ObsSnapshotInst         |
    | (image + SPICE state)   |
    +-------------------------+
                |
                v
    +-------------------------+
    | NavOrchestrator         |
    |   1. NavContext         |
    |   2. NavModels          |
    |   3. Features (gated)   |
    |   4. Pass 1 techniques  |
    |   5. Ensemble -> prior  |
    |   6. Pass 2 techniques  |
    |   7. Final ensemble     |
    +-------------------------+
                |
                v
    +-------------------------+
    | NavResult               |
    |   offset_px, sigma,     |
    |   confidence_rank,      |
    |   per_technique,        |
    |   annotations,          |
    |   provenance            |
    +-------------------------+
                |
                v
    +-------------------------+
    | navigate_image_files    |
    |   _metadata.json        |
    |   _summary.png          |
    +-------------------------+

The summary PNG renderer
========================

:func:`nav.navigate_image_files._write_summary_png` is a thin driver:

1. Apply a percentile-based linear stretch
   (:func:`nav.support.image.apply_linear_gamma_stretch`) to ``obs.data``
   and replicate the result across three channels for the grayscale
   background.
2. Call :meth:`nav.annotation.annotations.Annotations.combine` on the
   collection assembled by every NavModel's ``to_annotations`` (merged
   inside the orchestrator) at the best-fit ``offset_px``.
3. Replace every background pixel where the overlay carries any non-zero
   channel with the overlay color, and PNG-encode via Pillow.

When the orchestrator returns a failure (no features extracted, image
classifier short-circuit, etc.) the renderer still writes the source
image alone — this is the audit trail the operator inspects to
understand *why* the pipeline failed on that frame.

Stopping early: ``prepare(obs)``
================================

For interactive drivers (the manual-nav dialog, debuggers, ad-hoc
inspection scripts) the same pre-technique state is exposed via
:meth:`NavOrchestrator.prepare`, which returns
``(context, kept_features)`` after the image classifier, NavModel build,
feature extraction, and reliability gate run — but **without** running
any technique or short-circuiting on hard-failure image classes.  The
manual-nav helper :func:`nav.nav_technique.run_manual_nav` (and the
``nav_offset --manual`` CLI flag) build on this entry point.

Per-image regression test
=========================

The end-to-end flow is exercised by
``tests/integration/test_autonomous_nav.py``.  For each operator-curated
sidecar (under ``tests/integration/image_library/images/<class>/``) the
test:

- resolves the sidecar's ``image_url`` against ``PDS3_HOLDINGS_DIR``,
- runs :func:`navigate_image_files`,
- asserts ``status``, ``confidence_rank``, ``offset_px`` (within
  ``offset_uncertainty_px + 0.5 px`` slack), the
  ``primary_technique`` (highest confidence; tie-break on
  ``(-confidence, technique_name)`` ascending), and the
  ``techniques_must_run`` / ``techniques_must_skip`` set membership.

The companion ``tests/integration/test_baselines.py`` records exact
rounded ``(offset_dv_px, offset_du_px, confidence)`` per image; baseline
mismatches require explicit operator review on the PR.

See :doc:`user_guide_image_library` for sidecar curation and the
"Save as Library Entry" workflow.
