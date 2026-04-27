"""Tests for ``nav.nav_orchestrator.nav_context.NavContext``."""

import numpy as np

from nav.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from nav.nav_orchestrator.nav_context import NavContext
from nav.nav_orchestrator.provenance import Provenance


def _minimal_context() -> NavContext:
    """Build a context with the smallest valid set of fields."""
    image = np.zeros((4, 4), np.float64)
    mask = np.ones((4, 4), bool)
    classifier = NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        noise_sigma=1.0,
        max_dn=10.0,
    )
    provenance = Provenance(
        rms_nav_version='0.5.2',
        image_et=0.0,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
    )
    return NavContext(
        obs=object(),
        image_ext=image,
        sensor_mask_ext=mask,
        image_noise_sigma=1.0,
        saturation_mask_ext=np.zeros((4, 4), bool),
        cosmic_ray_mask_ext=np.zeros((4, 4), bool),
        image_classifier=classifier,
        provenance=provenance,
    )


def test_navcontext_constructs_with_minimal_fields() -> None:
    """A minimal NavContext is constructible without optional fields."""
    ctx = _minimal_context()
    assert ctx.image_noise_sigma == 1.0
    assert ctx.prior_offset_px is None
    assert ctx.prior_covariance_px2 is None


def test_navcontext_with_prior_returns_new_instance() -> None:
    """``with_prior`` creates a new NavContext with the prior populated."""
    ctx = _minimal_context()
    cov = np.eye(2, dtype=np.float64) * 0.5
    new_ctx = ctx.with_prior(offset_px=(1.0, 2.0), covariance_px2=cov)
    assert new_ctx is not ctx
    assert new_ctx.prior_offset_px == (1.0, 2.0)
    assert ctx.prior_offset_px is None  # original unchanged


def test_navcontext_with_prior_preserves_other_fields() -> None:
    """``with_prior`` only changes the prior fields."""
    ctx = _minimal_context()
    cov = np.eye(2, dtype=np.float64)
    new_ctx = ctx.with_prior(offset_px=(0.0, 0.0), covariance_px2=cov)
    assert new_ctx.image_noise_sigma == ctx.image_noise_sigma
    assert new_ctx.image_classifier is ctx.image_classifier
