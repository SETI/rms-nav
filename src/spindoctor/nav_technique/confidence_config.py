"""Build :class:`ConfidenceSpec` instances from the techniques config block.

The per-technique confidence-formula coefficients live in
``config_files/config_510_techniques.yaml`` under the ``techniques`` key.
Each technique reads its own block via :func:`load_confidence_spec`,
which validates the YAML shape and returns a frozen
:class:`ConfidenceSpec`.  Mis-typed numbers, unknown keys, and missing
``alpha0`` raise at config-load time so a malformed file fails fast at
process startup instead of mid-image.
"""

from __future__ import annotations

import math
from typing import Any

from spindoctor.nav_technique.confidence import ConfidenceSpec, ConfidenceTerm

__all__ = ['ConfidenceConfigError', 'load_confidence_spec', 'load_technique_tuning']


_VALID_TERM_KEYS: frozenset[str] = frozenset({'feature', 'alpha', 'offset', 'divisor', 'cap_at'})
_VALID_BLOCK_KEYS: frozenset[str] = frozenset(
    {'alpha0', 'terms', 'hard_zero_if', 'hard_cap', 'tuning'}
)


class ConfidenceConfigError(ValueError):
    """Raised when ``config_510_techniques.yaml`` cannot build a valid spec.

    Always names the technique whose block triggered the failure so a
    typo or shape error surfaces with full diagnostic at startup.
    """


def load_confidence_spec(techniques: dict[str, Any], technique_name: str) -> ConfidenceSpec:
    """Return the :class:`ConfidenceSpec` for ``technique_name``.

    Reads the ``techniques[technique_name]`` mapping and constructs a
    :class:`ConfidenceSpec` from its fields.  All numeric coercion and
    range checks happen here so per-technique modules stay free of
    YAML-shape bookkeeping.  Callers typically pass
    ``Config.category('techniques')`` (an ``AttrDict``, which subclasses
    ``dict``).

    Parameters:
        techniques: Mapping of ``NavTechnique.name`` to its raw config
            block (the ``techniques`` section of
            ``config_510_techniques.yaml``).
        technique_name: ``NavTechnique.name`` to look up.

    Returns:
        A frozen :class:`ConfidenceSpec` ready to assign to the
        technique's ``confidence_spec`` class attribute.

    Raises:
        ConfidenceConfigError: If the block is missing, has the wrong
            shape, or carries an unknown / malformed field.
    """
    techniques = _require_mapping(techniques, 'techniques')
    if technique_name not in techniques:
        raise ConfidenceConfigError(
            f'config_510_techniques.yaml: missing block for technique '
            f'{technique_name!r}; declared techniques are '
            f'{sorted(techniques.keys())!r}'
        )
    block = _require_mapping(techniques[technique_name], f'techniques.{technique_name}')
    unknown = set(block.keys()) - _VALID_BLOCK_KEYS
    if unknown:
        raise ConfidenceConfigError(
            f'techniques.{technique_name}: unknown keys {sorted(unknown)!r}; '
            f'valid keys are {sorted(_VALID_BLOCK_KEYS)!r}'
        )
    if 'alpha0' not in block:
        raise ConfidenceConfigError(f'techniques.{technique_name}: missing required key alpha0')
    alpha0 = _require_finite_float(block['alpha0'], f'techniques.{technique_name}.alpha0')
    raw_terms = block.get('terms', [])
    if not isinstance(raw_terms, list):
        raise ConfidenceConfigError(
            f'techniques.{technique_name}.terms: expected a list, got {type(raw_terms).__name__}'
        )
    terms = tuple(
        _build_term(entry, location=f'techniques.{technique_name}.terms[{i}]')
        for i, entry in enumerate(raw_terms)
    )
    raw_gates = block.get('hard_zero_if', {})
    if not isinstance(raw_gates, dict):
        raise ConfidenceConfigError(
            f'techniques.{technique_name}.hard_zero_if: expected a mapping, got '
            f'{type(raw_gates).__name__}'
        )
    hard_zero_if: dict[str, bool] = {}
    for key, value in raw_gates.items():
        if not isinstance(key, str):
            raise ConfidenceConfigError(
                f'techniques.{technique_name}.hard_zero_if: keys must be strings, got '
                f'{type(key).__name__}'
            )
        if not isinstance(value, bool):
            raise ConfidenceConfigError(
                f'techniques.{technique_name}.hard_zero_if[{key!r}]: must be bool, got '
                f'{type(value).__name__}'
            )
        hard_zero_if[key] = value
    raw_cap = block.get('hard_cap')
    hard_cap = (
        None
        if raw_cap is None
        else _require_finite_float(raw_cap, f'techniques.{technique_name}.hard_cap')
    )
    try:
        return ConfidenceSpec(
            alpha0=alpha0,
            terms=terms,
            hard_zero_if=hard_zero_if,
            hard_cap=hard_cap,
        )
    except (TypeError, ValueError) as exc:
        raise ConfidenceConfigError(
            f'techniques.{technique_name}: ConfidenceSpec construction failed: {exc}'
        ) from exc


def load_technique_tuning(
    techniques: dict[str, Any], technique_name: str
) -> dict[str, float | int]:
    """Return the ``tuning`` block for ``technique_name``.

    Returns the dict-shaped ``tuning`` sub-block as a flat
    ``{key: number}`` mapping with all numeric values coerced to
    ``float`` or ``int`` (booleans rejected — YAML's loose numeric
    types make ``True`` look numeric without this guard).  Each
    technique consumer pulls the values it needs by name and supplies
    its own default for missing keys.

    Parameters:
        techniques: Mapping of ``NavTechnique.name`` to its raw config
            block (the ``techniques`` section of
            ``config_510_techniques.yaml``).
        technique_name: ``NavTechnique.name`` to look up.

    Returns:
        ``{key: value}`` or an empty mapping when the technique block
        has no ``tuning`` sub-block.

    Raises:
        ConfidenceConfigError: If the block is malformed or any
            tuning value is not a finite numeric scalar.
    """
    techniques = _require_mapping(techniques, 'techniques')
    if technique_name not in techniques:
        return {}
    block = _require_mapping(techniques[technique_name], f'techniques.{technique_name}')
    raw = block.get('tuning', {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfidenceConfigError(
            f'techniques.{technique_name}.tuning: expected a mapping, got {type(raw).__name__}'
        )
    out: dict[str, float | int] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ConfidenceConfigError(
                f'techniques.{technique_name}.tuning: keys must be strings, got '
                f'{type(key).__name__}'
            )
        location = f'techniques.{technique_name}.tuning.{key}'
        if isinstance(value, bool):
            raise ConfidenceConfigError(f'{location}: must be numeric, got bool {value!r}')
        if isinstance(value, int):
            out[key] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ConfidenceConfigError(f'{location}: must be finite, got {value!r}')
            out[key] = value
        else:
            raise ConfidenceConfigError(f'{location}: must be numeric, got {type(value).__name__}')
    return out


def _build_term(entry: Any, *, location: str) -> ConfidenceTerm:
    """Construct one :class:`ConfidenceTerm` from a YAML mapping entry."""
    if not isinstance(entry, dict):
        raise ConfidenceConfigError(
            f'{location}: each terms entry must be a mapping, got {type(entry).__name__}'
        )
    unknown = set(entry.keys()) - _VALID_TERM_KEYS
    if unknown:
        raise ConfidenceConfigError(
            f'{location}: unknown keys {sorted(unknown)!r}; valid keys are '
            f'{sorted(_VALID_TERM_KEYS)!r}'
        )
    if 'feature' not in entry:
        raise ConfidenceConfigError(f'{location}: missing required key feature')
    if 'alpha' not in entry:
        raise ConfidenceConfigError(f'{location}: missing required key alpha')
    feature = entry['feature']
    if not isinstance(feature, str) or not feature.strip():
        raise ConfidenceConfigError(
            f'{location}.feature: must be a non-empty string, got {feature!r}'
        )
    alpha = _require_finite_float(entry['alpha'], f'{location}.alpha')
    offset = (
        0.0
        if entry.get('offset') is None
        else _require_finite_float(entry['offset'], f'{location}.offset')
    )
    divisor = (
        1.0
        if entry.get('divisor') is None
        else _require_finite_float(entry['divisor'], f'{location}.divisor')
    )
    cap_at = (
        None
        if entry.get('cap_at') is None
        else _require_finite_float(entry['cap_at'], f'{location}.cap_at')
    )
    try:
        return ConfidenceTerm(
            feature=feature,
            alpha=alpha,
            offset=offset,
            divisor=divisor,
            cap_at=cap_at,
        )
    except (TypeError, ValueError) as exc:
        raise ConfidenceConfigError(f'{location}: {exc}') from exc


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    """Coerce ``value`` to ``dict[str, Any]`` or raise a typed error."""
    if not isinstance(value, dict):
        raise ConfidenceConfigError(f'{location}: expected a mapping, got {type(value).__name__}')
    return value


def _require_finite_float(value: Any, location: str) -> float:
    """Coerce ``value`` to ``float`` or raise a typed error."""
    if isinstance(value, bool):
        raise ConfidenceConfigError(f'{location}: must be numeric, got bool {value!r}')
    if not isinstance(value, (int, float)):
        raise ConfidenceConfigError(f'{location}: must be numeric, got {type(value).__name__}')
    coerced = float(value)
    if not math.isfinite(coerced):
        raise ConfidenceConfigError(f'{location}: must be finite, got {value!r}')
    return coerced
