"""Spec tests for ``spindoctor.reproj._context_managers._reduced_oops_precision``.

The documented contract (module docstring, function docstring, and the dev guide's
"Context managers" section) is: the manager sets ``oops.config.PATH_PHOTONS.dlt_precision``
and ``oops.config.SURFACE_PHOTONS.dlt_precision`` to ``dlt`` (default 1) for the
duration of the block and restores both original values on exit, even when the
wrapped code raises.
"""

from collections.abc import Iterator

import oops
import pytest

from spindoctor.reproj._context_managers import _reduced_oops_precision


@pytest.fixture(autouse=True)
def _oops_precision_guard() -> Iterator[None]:
    """Snapshot and restore the oops global precision attributes around each test.

    Yields:
        None; the pre-test values are restored afterwards even if a test fails.
    """
    old_path = oops.config.PATH_PHOTONS.dlt_precision
    old_surf = oops.config.SURFACE_PHOTONS.dlt_precision
    yield
    oops.config.PATH_PHOTONS.dlt_precision = old_path
    oops.config.SURFACE_PHOTONS.dlt_precision = old_surf


class TestReducedOopsPrecision:
    """Set-and-restore contract of ``_reduced_oops_precision``."""

    def test_default_sets_path_photons_to_one(self) -> None:
        """Inside the block, PATH_PHOTONS.dlt_precision is the documented default of 1."""
        with _reduced_oops_precision():
            assert oops.config.PATH_PHOTONS.dlt_precision == 1

    def test_default_sets_surface_photons_to_one(self) -> None:
        """Inside the block, SURFACE_PHOTONS.dlt_precision is the documented default of 1."""
        with _reduced_oops_precision():
            assert oops.config.SURFACE_PHOTONS.dlt_precision == 1

    def test_restores_path_photons_on_normal_exit(self) -> None:
        """PATH_PHOTONS.dlt_precision is restored to its prior value after the block."""
        before = oops.config.PATH_PHOTONS.dlt_precision
        with _reduced_oops_precision():
            pass
        assert oops.config.PATH_PHOTONS.dlt_precision == before

    def test_restores_surface_photons_on_normal_exit(self) -> None:
        """SURFACE_PHOTONS.dlt_precision is restored to its prior value after the block."""
        before = oops.config.SURFACE_PHOTONS.dlt_precision
        with _reduced_oops_precision():
            pass
        assert oops.config.SURFACE_PHOTONS.dlt_precision == before

    def test_custom_dlt_applied_to_path_photons(self) -> None:
        """An explicit dlt value is applied to PATH_PHOTONS."""
        with _reduced_oops_precision(dlt=7):
            assert oops.config.PATH_PHOTONS.dlt_precision == 7

    def test_custom_dlt_applied_to_surface_photons(self) -> None:
        """An explicit dlt value is applied to SURFACE_PHOTONS."""
        with _reduced_oops_precision(dlt=7):
            assert oops.config.SURFACE_PHOTONS.dlt_precision == 7

    def test_restores_path_photons_on_exception(self) -> None:
        """PATH_PHOTONS.dlt_precision is restored even when the block raises."""
        before = oops.config.PATH_PHOTONS.dlt_precision
        with pytest.raises(RuntimeError, match='boom'), _reduced_oops_precision():
            raise RuntimeError('boom')
        assert oops.config.PATH_PHOTONS.dlt_precision == before

    def test_restores_surface_photons_on_exception(self) -> None:
        """SURFACE_PHOTONS.dlt_precision is restored even when the block raises."""
        before = oops.config.SURFACE_PHOTONS.dlt_precision
        with pytest.raises(RuntimeError, match='boom'), _reduced_oops_precision():
            raise RuntimeError('boom')
        assert oops.config.SURFACE_PHOTONS.dlt_precision == before

    def test_restores_non_default_prior_values(self) -> None:
        """Restoration returns to whatever values were in effect, not to library defaults."""
        oops.config.PATH_PHOTONS.dlt_precision = 0.125
        oops.config.SURFACE_PHOTONS.dlt_precision = 0.25
        with _reduced_oops_precision():
            assert oops.config.PATH_PHOTONS.dlt_precision == 1
        assert oops.config.PATH_PHOTONS.dlt_precision == 0.125
        assert oops.config.SURFACE_PHOTONS.dlt_precision == 0.25

    def test_nested_blocks_restore_in_lifo_order(self) -> None:
        """Exiting an inner block restores the outer block's value, then the original."""
        original = oops.config.PATH_PHOTONS.dlt_precision
        with _reduced_oops_precision(dlt=1):
            with _reduced_oops_precision(dlt=5):
                assert oops.config.PATH_PHOTONS.dlt_precision == 5
                assert oops.config.SURFACE_PHOTONS.dlt_precision == 5
            assert oops.config.PATH_PHOTONS.dlt_precision == 1
            assert oops.config.SURFACE_PHOTONS.dlt_precision == 1
        assert oops.config.PATH_PHOTONS.dlt_precision == original

    def test_yields_none(self) -> None:
        """The context manager yields None (no handle is exposed)."""
        with _reduced_oops_precision() as handle:
            assert handle is None

    def test_dlt_is_keyword_only(self) -> None:
        """The dlt parameter cannot be passed positionally."""
        with pytest.raises(TypeError, match='positional'), _reduced_oops_precision(2):  # type: ignore[misc]
            pass
