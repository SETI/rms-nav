"""Shared sigmoid-of-linear-combination confidence formula.

Every NavTechnique converts its ``NavTechniqueDiagnostics`` into a
calibrated [0, 1] confidence using the same shape:

    confidence = sigmoid(alpha0 + sum_i alpha_i * normalize_i(x_i))

where ``normalize_i`` applies the spec's ``offset`` -> ``divisor`` -> ``cap_at``
transformation.  Hard-zero gates (e.g. ``at_edge=True``) force confidence
to 0; an optional ``hard_cap`` clamps the result.

The math lives here so techniques' YAML formula specs can be evaluated
uniformly and a config-load validation pass can verify every spec at
startup.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    'ConfidenceSpec',
    'ConfidenceTerm',
    'evaluate_sigmoid_combination',
]

logging.getLogger(__name__).addHandler(logging.NullHandler())


_NumberType = (int, float)


@dataclass(frozen=True)
class ConfidenceTerm:
    """One term in the sigmoid-of-linear-combination formula.

    Parameters:
        feature: Name of the diagnostic-dataclass attribute supplying the
            raw value.
        alpha: Linear-combination coefficient.
        offset: Subtracted from the raw value before scaling.
        divisor: Raw value is divided by this after offset (default 1).
            Must be non-zero.
        cap_at: Optional upper bound on the post-scale value (clamped to
            ``[0, cap_at]`` if set).  When set, must lie in ``[0, 1]``.

    Raises:
        TypeError: if ``alpha``, ``offset``, ``divisor``, or ``cap_at``
            is not numeric.
        ValueError: if ``divisor`` is zero or ``cap_at`` is outside
            ``[0, 1]``.
    """

    feature: str
    alpha: float
    offset: float = 0.0
    divisor: float = 1.0
    cap_at: float | None = None

    def __post_init__(self) -> None:
        """Validate numeric types and divisor / cap_at ranges."""
        for name in ('alpha', 'offset', 'divisor'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, _NumberType):
                raise TypeError(
                    f'ConfidenceTerm.{name} must be numeric; got {type(value).__name__}'
                )
        if self.divisor == 0.0:
            raise ValueError('ConfidenceTerm.divisor must be non-zero')
        if self.cap_at is not None:
            if isinstance(self.cap_at, bool) or not isinstance(self.cap_at, _NumberType):
                raise TypeError(
                    f'ConfidenceTerm.cap_at must be numeric or None; got '
                    f'{type(self.cap_at).__name__}'
                )
            if not 0.0 <= self.cap_at <= 1.0:
                raise ValueError(f'ConfidenceTerm.cap_at must lie in [0, 1]; got {self.cap_at!r}')


@dataclass(frozen=True)
class ConfidenceSpec:
    """Full confidence formula spec for one NavTechnique.

    Parameters:
        alpha0: Constant term in the sigmoid argument.
        terms: Tuple of ``ConfidenceTerm`` linear contributions.
        hard_zero_if: Mapping of diagnostic-attribute names (str) to
            expected boolean values; if any condition holds,
            confidence = 0.
        hard_cap: Optional upper-bound clamp applied after the sigmoid.
            When set, must lie in ``[0, 1]``.

    Raises:
        TypeError: if ``alpha0`` is not numeric, ``terms`` is not a
            tuple of ``ConfidenceTerm``, or ``hard_zero_if`` is not a
            mapping of ``str`` to ``bool``.
        ValueError: if ``hard_cap`` is outside ``[0, 1]``.
    """

    alpha0: float
    terms: tuple[ConfidenceTerm, ...] = ()
    hard_zero_if: dict[str, bool] = field(default_factory=dict)
    hard_cap: float | None = None

    def __post_init__(self) -> None:
        """Validate types and ranges of every field."""
        if isinstance(self.alpha0, bool) or not isinstance(self.alpha0, _NumberType):
            raise TypeError(
                f'ConfidenceSpec.alpha0 must be numeric; got {type(self.alpha0).__name__}'
            )
        if not isinstance(self.terms, tuple):
            raise TypeError(
                f'ConfidenceSpec.terms must be a tuple; got {type(self.terms).__name__}'
            )
        for term in self.terms:
            if not isinstance(term, ConfidenceTerm):
                raise TypeError(
                    'ConfidenceSpec.terms entries must be ConfidenceTerm; '
                    f'got {type(term).__name__}'
                )
        if not isinstance(self.hard_zero_if, dict):
            raise TypeError(
                'ConfidenceSpec.hard_zero_if must be a dict; '
                f'got {type(self.hard_zero_if).__name__}'
            )
        for key, value in self.hard_zero_if.items():
            if not isinstance(key, str):
                raise TypeError(
                    f'ConfidenceSpec.hard_zero_if keys must be str; got {type(key).__name__}'
                )
            if not isinstance(value, bool):
                raise TypeError(
                    f'ConfidenceSpec.hard_zero_if[{key!r}] must be bool; got {type(value).__name__}'
                )
        if self.hard_cap is not None:
            if isinstance(self.hard_cap, bool) or not isinstance(self.hard_cap, _NumberType):
                raise TypeError(
                    f'ConfidenceSpec.hard_cap must be numeric or None; got '
                    f'{type(self.hard_cap).__name__}'
                )
            if not 0.0 <= self.hard_cap <= 1.0:
                raise ValueError(
                    f'ConfidenceSpec.hard_cap must lie in [0, 1]; got {self.hard_cap!r}'
                )


def _sigmoid(x: float) -> float:
    """Numerically-stable logistic sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _normalize(value: float, term: ConfidenceTerm) -> float:
    """Apply offset / divisor / cap_at to a raw diagnostic value."""
    scaled = (value - term.offset) / term.divisor
    if term.cap_at is not None:
        if scaled < 0.0:
            scaled = 0.0
        elif scaled > term.cap_at:
            scaled = term.cap_at
    return scaled


def evaluate_sigmoid_combination(
    spec: ConfidenceSpec, diagnostics: Any, *, technique_name: str = ''
) -> float:
    """Evaluate the spec's sigmoid formula against a diagnostics object.

    Parameters:
        spec: The technique's confidence formula spec.
        diagnostics: Diagnostics dataclass instance for the technique.
        technique_name: Optional human-readable identifier used in error
            messages.

    Returns:
        Calibrated confidence in ``[0, 1]``.

    Raises:
        ValueError: if a term references an attribute that does not exist
            on the diagnostics object.  The message names the missing
            attribute and the technique.
    """
    # Hard-zero gates first: short-circuit if any condition holds.
    for attr_name, required in spec.hard_zero_if.items():
        if not hasattr(diagnostics, attr_name):
            raise ValueError(
                f'confidence spec {technique_name!r}: hard_zero_if attribute '
                f'{attr_name!r} not found on diagnostics '
                f'{type(diagnostics).__name__}'
            )
        actual = getattr(diagnostics, attr_name)
        if bool(actual) == bool(required):
            return 0.0
    # Linear-combination of normalized terms.
    arg = spec.alpha0
    for term in spec.terms:
        if not hasattr(diagnostics, term.feature):
            raise ValueError(
                f'confidence spec {technique_name!r}: feature attribute '
                f'{term.feature!r} not found on diagnostics '
                f'{type(diagnostics).__name__}'
            )
        raw = float(getattr(diagnostics, term.feature))
        arg += term.alpha * _normalize(raw, term)
    confidence = _sigmoid(arg)
    if spec.hard_cap is not None and confidence > spec.hard_cap:
        confidence = spec.hard_cap
    return confidence
