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

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    'ConfidenceBreakdown',
    'ConfidenceSpec',
    'ConfidenceTerm',
    'ConfidenceTermContribution',
    'evaluate_sigmoid_combination',
]

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
        if not isinstance(self.feature, str):
            raise TypeError(
                f'ConfidenceTerm.feature must be str; got {type(self.feature).__name__}'
            )
        if not self.feature.strip():
            raise ValueError('ConfidenceTerm.feature must be a non-empty string')
        for name in ('alpha', 'offset', 'divisor'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, _NumberType):
                raise TypeError(
                    f'ConfidenceTerm.{name} must be numeric; got {type(value).__name__}'
                )
            if not math.isfinite(value):
                raise ValueError(f'ConfidenceTerm.{name} must be finite; got {value!r}')
        if self.divisor == 0.0:
            raise ValueError('ConfidenceTerm.divisor must be non-zero')
        if self.cap_at is not None:
            if isinstance(self.cap_at, bool) or not isinstance(self.cap_at, _NumberType):
                raise TypeError(
                    f'ConfidenceTerm.cap_at must be numeric or None; got '
                    f'{type(self.cap_at).__name__}'
                )
            if not math.isfinite(self.cap_at):
                raise ValueError(f'ConfidenceTerm.cap_at must be finite; got {self.cap_at!r}')
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
        # Defensive shallow copy so the caller cannot mutate the
        # frozen dataclass's hard_zero_if dict after construction.
        object.__setattr__(self, 'hard_zero_if', dict(self.hard_zero_if))
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


@dataclass(frozen=True)
class ConfidenceTermContribution:
    """One term's contribution to the sigmoid argument.

    Parameters:
        feature: The diagnostic-attribute name.
        raw: The raw value read off the diagnostics object.
        normalized: ``raw`` after offset / divisor / cap_at.
        alpha: Linear coefficient applied.
        contribution: ``alpha * normalized`` — the term's signed
            contribution to the sigmoid argument.
    """

    feature: str
    raw: float
    normalized: float
    alpha: float
    contribution: float


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Per-step trace of a sigmoid-combination evaluation.

    Parameters:
        confidence: Final ``[0, 1]`` confidence after sigmoid + hard_cap.
        sigmoid_arg: The argument fed into the sigmoid (sum of alpha0 +
            term contributions).
        alpha0: The constant baseline term.
        terms: Per-term contributions.
        hard_zero: Name of the ``hard_zero_if`` attribute that fired, or
            ``None`` when no hard-zero gate applied.
        hard_cap_applied: True if the final sigmoid value was clamped by
            ``spec.hard_cap``.
    """

    confidence: float
    sigmoid_arg: float
    alpha0: float
    terms: tuple[ConfidenceTermContribution, ...]
    hard_zero: str | None
    hard_cap_applied: bool


def evaluate_sigmoid_combination(
    spec: ConfidenceSpec,
    diagnostics: Any,
    *,
    technique_name: str = '',
    return_breakdown: bool = False,
) -> Any:
    """Evaluate the spec's sigmoid formula against a diagnostics object.

    Parameters:
        spec: The technique's confidence formula spec.
        diagnostics: Diagnostics dataclass instance for the technique.
        technique_name: Optional human-readable identifier used in error
            messages.
        return_breakdown: When True, return a tuple
            ``(confidence, ConfidenceBreakdown)`` so callers can log a
            per-term explanation of low / zero confidence values.

    Returns:
        Calibrated confidence in ``[0, 1]``, or
        ``(confidence, ConfidenceBreakdown)`` when
        ``return_breakdown=True``.

    Raises:
        ValueError: if a term references an attribute that does not exist
            on the diagnostics object.  The message names the missing
            attribute and the technique.
    """
    contributions: list[ConfidenceTermContribution] = []
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
            if not return_breakdown:
                return 0.0
            breakdown = ConfidenceBreakdown(
                confidence=0.0,
                sigmoid_arg=float('nan'),
                alpha0=spec.alpha0,
                terms=tuple(contributions),
                hard_zero=attr_name,
                hard_cap_applied=False,
            )
            return 0.0, breakdown
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
        normalized = _normalize(raw, term)
        contribution = term.alpha * normalized
        arg += contribution
        contributions.append(
            ConfidenceTermContribution(
                feature=term.feature,
                raw=raw,
                normalized=normalized,
                alpha=term.alpha,
                contribution=contribution,
            )
        )
    confidence = _sigmoid(arg)
    hard_cap_applied = False
    if spec.hard_cap is not None and confidence > spec.hard_cap:
        confidence = spec.hard_cap
        hard_cap_applied = True
    if not return_breakdown:
        return confidence
    breakdown = ConfidenceBreakdown(
        confidence=confidence,
        sigmoid_arg=arg,
        alpha0=spec.alpha0,
        terms=tuple(contributions),
        hard_zero=None,
        hard_cap_applied=hard_cap_applied,
    )
    return confidence, breakdown
