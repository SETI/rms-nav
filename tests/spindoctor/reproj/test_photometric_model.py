"""Spec-first unit tests for spindoctor.reproj.photometric_model.

Contracts under test come from the class and method docstrings in
``src/spindoctor/reproj/photometric_model.py`` and the "Photometric models"
section of ``docs/dev_guide/dev_guide_reprojection.rst``: all angles are in
radians, Lambert divides by a clamped cos(incidence), Lommel-Seeliger applies
``(mu0 + mu) / (2 * mu0)`` with a signed denominator floor, Minnaert divides by
``cos(incidence)^k * cos(emission)^(k-1)``, every ``uncorrect`` inverts its
``correct`` with identical clamping, and ``photometric_model_from_name``
resolves case/space/hyphen-insensitive aliases.
"""

import math

import numpy as np
import numpy.typing as npt
import pytest

from spindoctor.reproj.photometric_model import (
    LambertModel,
    LommelSeeligerModel,
    MinnaertModel,
    photometric_model_from_name,
)

_DATA = np.array([[1.0, 2.0], [3.0, 4.0]])


def _angles(value: float) -> npt.NDArray[np.float64]:
    """Return a 2x2 constant angle array matching ``_DATA``.

    Parameters:
        value: Angle in radians to fill the array with.
    """
    return np.full((2, 2), value)


# =========================================================================
# Lambert
# =========================================================================


class TestLambertModel:
    """LambertModel divides by max(cos(incidence), min_cos_incidence)."""

    def test_name(self) -> None:
        """The model name is 'lambert'."""
        assert LambertModel().name == 'lambert'

    def test_zero_incidence_is_identity(self) -> None:
        """cos(0) == 1, so normal illumination leaves the data unchanged."""
        out = LambertModel().correct(
            _DATA, incidence=_angles(0.0), emission=_angles(0.3), phase=_angles(0.5)
        )
        np.testing.assert_allclose(out, _DATA)

    def test_divides_by_cos_incidence(self) -> None:
        """Moderate incidence divides the data by cos(incidence)."""
        inc = 1.0
        out = LambertModel().correct(
            _DATA, incidence=_angles(inc), emission=_angles(0.3), phase=_angles(0.5)
        )
        np.testing.assert_allclose(out, _DATA / math.cos(inc))

    def test_emission_and_phase_are_ignored(self) -> None:
        """Changing emission and phase does not change the Lambert correction."""
        a = LambertModel().correct(
            _DATA, incidence=_angles(0.4), emission=_angles(0.1), phase=_angles(0.1)
        )
        b = LambertModel().correct(
            _DATA, incidence=_angles(0.4), emission=_angles(1.4), phase=_angles(2.8)
        )
        np.testing.assert_allclose(a, b)

    def test_grazing_incidence_uses_clamp(self) -> None:
        """Near-grazing incidence divides by min_cos_incidence, not by ~0."""
        out = LambertModel(min_cos_incidence=0.01).correct(
            _DATA, incidence=_angles(math.pi / 2.0), emission=_angles(0.3), phase=_angles(0.5)
        )
        np.testing.assert_allclose(out, _DATA / 0.01)

    def test_uncorrect_inverts_correct(self) -> None:
        """uncorrect(correct(x)) == x at moderate angles."""
        model = LambertModel()
        inc = _angles(0.7)
        emi = _angles(0.3)
        pha = _angles(0.5)
        corrected = model.correct(_DATA, incidence=inc, emission=emi, phase=pha)
        out = model.uncorrect(corrected, incidence=inc, emission=emi, phase=pha)
        np.testing.assert_allclose(out, _DATA)

    def test_uncorrect_uses_same_clamp(self) -> None:
        """uncorrect at grazing incidence multiplies by the clamp value."""
        out = LambertModel(min_cos_incidence=0.01).uncorrect(
            _DATA, incidence=_angles(math.pi / 2.0), emission=_angles(0.3), phase=_angles(0.5)
        )
        np.testing.assert_allclose(out, _DATA * 0.01)

    def test_bool_clamp_raises_type_error(self) -> None:
        """A bool min_cos_incidence is rejected."""
        with pytest.raises(TypeError, match='min_cos_incidence must be a real number'):
            LambertModel(min_cos_incidence=True)

    def test_zero_clamp_raises_value_error(self) -> None:
        """min_cos_incidence == 0 is outside the valid cosine clamp range."""
        with pytest.raises(ValueError, match='min_cos_incidence must be finite with'):
            LambertModel(min_cos_incidence=0.0)

    def test_clamp_above_one_raises_value_error(self) -> None:
        """min_cos_incidence > 1 is outside the valid cosine clamp range."""
        with pytest.raises(ValueError, match='min_cos_incidence must be finite with'):
            LambertModel(min_cos_incidence=1.5)

    def test_nan_clamp_raises_value_error(self) -> None:
        """A NaN min_cos_incidence is rejected as non-finite."""
        with pytest.raises(ValueError, match='min_cos_incidence must be finite'):
            LambertModel(min_cos_incidence=math.nan)


# =========================================================================
# Lommel-Seeliger
# =========================================================================


class TestLommelSeeligerModel:
    """LommelSeeligerModel multiplies by (mu0 + mu) / (2 * mu0)."""

    def test_name(self) -> None:
        """The model name is 'lommel_seeliger'."""
        assert LommelSeeligerModel().name == 'lommel_seeliger'

    def test_normal_geometry_is_identity(self) -> None:
        """At incidence == emission == 0 the factor is (1 + 1) / 2 == 1."""
        out = LommelSeeligerModel().correct(
            _DATA, incidence=_angles(0.0), emission=_angles(0.0), phase=_angles(0.5)
        )
        np.testing.assert_allclose(out, _DATA)

    def test_general_angles_follow_documented_formula(self) -> None:
        """correct() multiplies by (cos i + cos e) / (2 cos i) at moderate angles."""
        inc, emi = 0.4, 0.9
        out = LommelSeeligerModel().correct(
            _DATA, incidence=_angles(inc), emission=_angles(emi), phase=_angles(0.5)
        )
        factor = (math.cos(inc) + math.cos(emi)) / (2.0 * math.cos(inc))
        np.testing.assert_allclose(out, _DATA * factor)

    def test_phase_is_ignored(self) -> None:
        """Changing phase does not change the correction."""
        a = LommelSeeligerModel().correct(
            _DATA, incidence=_angles(0.4), emission=_angles(0.9), phase=_angles(0.1)
        )
        b = LommelSeeligerModel().correct(
            _DATA, incidence=_angles(0.4), emission=_angles(0.9), phase=_angles(2.9)
        )
        np.testing.assert_allclose(a, b)

    def test_grazing_incidence_uses_clamp(self) -> None:
        """cos(incidence) is clamped to min_cos_incidence at grazing incidence."""
        emi = 0.3
        out = LommelSeeligerModel(min_cos_incidence=0.01).correct(
            _DATA, incidence=_angles(math.pi / 2.0), emission=_angles(emi), phase=_angles(0.5)
        )
        factor = (0.01 + math.cos(emi)) / 0.02
        np.testing.assert_allclose(out, _DATA * factor, rtol=1e-12)

    def test_near_zero_denominator_stays_finite(self) -> None:
        """The signed denominator floor keeps the output finite when mu0 + mu ~ 0."""
        # Clamped cos(incidence) == 0.01 and cos(emission) == -0.01 cancel.
        out = LommelSeeligerModel().correct(
            _DATA,
            incidence=_angles(math.pi / 2.0),
            emission=_angles(math.acos(-0.01)),
            phase=_angles(0.5),
        )
        assert bool(np.isfinite(out).all())

    def test_uncorrect_inverts_correct(self) -> None:
        """uncorrect(correct(x)) == x at moderate angles."""
        model = LommelSeeligerModel()
        inc = _angles(0.6)
        emi = _angles(1.1)
        pha = _angles(0.5)
        corrected = model.correct(_DATA, incidence=inc, emission=emi, phase=pha)
        out = model.uncorrect(corrected, incidence=inc, emission=emi, phase=pha)
        np.testing.assert_allclose(out, _DATA)

    def test_min_denom_below_floor_raises_value_error(self) -> None:
        """min_denom below the smallest allowed positive value is rejected."""
        with pytest.raises(ValueError, match='min_denom must be finite and >='):
            LommelSeeligerModel(min_denom=0.0)

    def test_bool_min_denom_raises_type_error(self) -> None:
        """A bool min_denom is rejected."""
        with pytest.raises(TypeError, match='min_denom must be a real number'):
            LommelSeeligerModel(min_denom=False)


# =========================================================================
# Minnaert
# =========================================================================


class TestMinnaertModel:
    """MinnaertModel divides by cos(i)^k * cos(e)^(k-1)."""

    def test_name(self) -> None:
        """The model name is 'minnaert'."""
        assert MinnaertModel().name == 'minnaert'

    def test_k_one_reduces_to_lambert(self) -> None:
        """With k == 1 the Minnaert correction equals the Lambert correction."""
        inc = _angles(0.8)
        emi = _angles(0.4)
        pha = _angles(0.5)
        minnaert = MinnaertModel(k=1.0).correct(_DATA, incidence=inc, emission=emi, phase=pha)
        lambert = LambertModel().correct(_DATA, incidence=inc, emission=emi, phase=pha)
        np.testing.assert_allclose(minnaert, lambert)

    def test_default_k_half_follows_documented_formula(self) -> None:
        """With k == 0.5 the correction multiplies by sqrt(cos e / cos i)."""
        inc, emi = 0.7, 0.4
        out = MinnaertModel().correct(
            _DATA, incidence=_angles(inc), emission=_angles(emi), phase=_angles(0.5)
        )
        expected = _DATA / (math.cos(inc) ** 0.5 * math.cos(emi) ** (-0.5))
        np.testing.assert_allclose(out, expected)

    def test_both_cosines_clamped(self) -> None:
        """Grazing incidence and emission both use their configured clamps."""
        out = MinnaertModel(k=0.5, min_cos_incidence=0.02, min_cos_emission=0.04).correct(
            _DATA,
            incidence=_angles(math.pi / 2.0),
            emission=_angles(math.pi / 2.0),
            phase=_angles(0.5),
        )
        expected = _DATA / (0.02**0.5 * 0.04**-0.5)
        np.testing.assert_allclose(out, expected, rtol=1e-12)

    def test_uncorrect_inverts_correct(self) -> None:
        """uncorrect(correct(x)) == x at moderate angles."""
        model = MinnaertModel(k=0.7)
        inc = _angles(0.5)
        emi = _angles(0.9)
        pha = _angles(0.5)
        corrected = model.correct(_DATA, incidence=inc, emission=emi, phase=pha)
        out = model.uncorrect(corrected, incidence=inc, emission=emi, phase=pha)
        np.testing.assert_allclose(out, _DATA)

    def test_non_finite_k_raises_value_error(self) -> None:
        """An infinite k is rejected."""
        with pytest.raises(ValueError, match='k must be finite'):
            MinnaertModel(k=math.inf)

    def test_bool_k_raises_type_error(self) -> None:
        """A bool k is rejected."""
        with pytest.raises(TypeError, match='k must be a real number'):
            MinnaertModel(k=True)

    def test_bad_emission_clamp_raises_value_error(self) -> None:
        """min_cos_emission outside (0, 1] is rejected."""
        with pytest.raises(ValueError, match='min_cos_emission must be finite with'):
            MinnaertModel(min_cos_emission=-0.5)


# =========================================================================
# Protocol conformance
# =========================================================================


class TestProtocolConformance:
    """All implementations satisfy the PhotometricModel protocol shape."""

    @pytest.mark.parametrize(
        'model',
        [LambertModel(), LommelSeeligerModel(), MinnaertModel()],
        ids=['lambert', 'lommel_seeliger', 'minnaert'],
    )
    def test_correct_preserves_shape(
        self, model: LambertModel | LommelSeeligerModel | MinnaertModel
    ) -> None:
        """correct() returns an array of the same shape as its input data.

        Parameters:
            model: Photometric model instance under test.
        """
        data = np.linspace(0.5, 2.0, 12).reshape(3, 4)
        out = model.correct(
            data,
            incidence=np.full((3, 4), 0.4),
            emission=np.full((3, 4), 0.3),
            phase=np.full((3, 4), 0.5),
        )
        assert out.shape == (3, 4)

    @pytest.mark.parametrize(
        'model',
        [LambertModel(), LommelSeeligerModel(), MinnaertModel()],
        ids=['lambert', 'lommel_seeliger', 'minnaert'],
    )
    def test_name_is_nonempty_string(
        self, model: LambertModel | LommelSeeligerModel | MinnaertModel
    ) -> None:
        """Every model exposes a non-empty string name.

        Parameters:
            model: Photometric model instance under test.
        """
        assert isinstance(model.name, str)
        assert len(model.name) > 0


# =========================================================================
# photometric_model_from_name
# =========================================================================


class TestPhotometricModelFromName:
    """Alias resolution and error behavior of photometric_model_from_name."""

    @pytest.mark.parametrize('name', [None, '', '  ', 'none', 'NULL', ' None '])
    def test_none_like_names_return_none(self, name: str | None) -> None:
        """None and none-like labels resolve to no model.

        Parameters:
            name: Stored model label to resolve.
        """
        assert photometric_model_from_name(name) is None

    def test_lambert_alias(self) -> None:
        """'lambert' (any case) resolves to LambertModel."""
        model = photometric_model_from_name('Lambert')
        assert isinstance(model, LambertModel)

    @pytest.mark.parametrize('name', ['lommel_seeliger', 'Lommel-Seeliger', 'lommelseeliger'])
    def test_lommel_seeliger_aliases(self, name: str) -> None:
        """Hyphen, underscore, and fused spellings all resolve to LommelSeeligerModel.

        Parameters:
            name: Stored model label to resolve.
        """
        model = photometric_model_from_name(name)
        assert isinstance(model, LommelSeeligerModel)

    def test_lommel_seeliger_space_alias(self) -> None:
        """A space-separated spelling resolves to LommelSeeligerModel."""
        model = photometric_model_from_name('lommel seeliger')
        assert isinstance(model, LommelSeeligerModel)

    def test_minnaert_alias(self) -> None:
        """'MINNAERT' with surrounding spaces resolves to MinnaertModel."""
        model = photometric_model_from_name('  MINNAERT ')
        assert isinstance(model, MinnaertModel)

    def test_unknown_name_raises_value_error(self) -> None:
        """An unrecognized label raises ValueError naming the input."""
        with pytest.raises(ValueError, match="Unknown photometric model name 'hapke'"):
            photometric_model_from_name('hapke')

    def test_returned_model_name_round_trips(self) -> None:
        """The name of a resolved model resolves back to the same model type."""
        model = photometric_model_from_name('lambert')
        assert model is not None
        again = photometric_model_from_name(model.name)
        assert type(again) is type(model)
