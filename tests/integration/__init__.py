"""Integration tests against the operator-curated image library.

The library lives under ``tests/integration/image_library/images/<class>/<image_id>.yaml``
and is the registry of operator-verified ground-truth offsets used to
calibrate and regression-test the autonomous navigator end-to-end.

Two test layers consume the library:

- ``test_image_library`` enforces structural invariants on the sidecars
  themselves; runs in the fast suite without holdings access.
- ``test_autonomous_nav`` runs the full orchestrator against the real
  PDS3 holdings; gated by the ``integration`` marker and skipped when
  ``PDS3_HOLDINGS_DIR`` is unset.
"""
