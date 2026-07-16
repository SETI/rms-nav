"""The artifact-mode registry: the single source of truth for scene defects.

The ``artifacts`` scene block is keyed, besides ``instrument_defaults`` and
``adversarial``, by the artifact-mode names in this registry.  Each mode is
described once here -- its rendering stage (``telemetry`` or ``detector``), its
parameter schema, the instruments it is available on, and whether it is
implemented yet -- and every consumer (the scene validator, the telemetry
stage, the detector hot-pixel routing) reads that description rather than
carrying its own copy.  Registering a mode is therefore the whole job of adding
one: a detector-stage mode that is unimplemented today drops in by flipping its
``implemented`` flag and adding its rendering code, with no change to the
validator or the block schema.

**Availability.**  A mode lists the sim instruments it is available on; a scene
that names the mode on any other instrument fails validation with a clear
message (the LORRI hot-pixel case carries a bespoke one, since LORRI has no hot
pixels by construction).  The instrument-agnostic ``generic`` / ``sim`` block
accepts every mode, which keeps unit scenes free to exercise any shape.

**Incidence.**  Every mode takes an ``incidence`` parameter, disabled at 0 (the
stage-activation rule: an incidence of 0, which is also the default even under
``instrument_defaults``, renders the mode as a no-op).  Its meaning is per-mode
and documented in each mode's ``incidence_semantics``: for count modes it is the
expected number of events (lost lines, blocks, spiked pixels) per frame, drawn
Poisson; for commanded / periodic modes it is the per-frame probability that the
mode activates at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    'ARTIFACT_MODES',
    'MODE_KEYS',
    'STRUCTURED_LOSS_ORDER',
    'ArtifactMode',
    'ModeParam',
    'mode_available',
    'mode_unavailable_message',
    'normalize_instrument',
    'resolve_mode_config',
]


# Instrument groupings used by the availability tables below.  ``generic`` is
# never listed: it is accepted unconditionally by mode_available.
_ALL_CCD: frozenset[str] = frozenset({'coiss_nac', 'coiss_wac', 'gossi', 'nhlorri'})
_ALL_INSTRUMENTS: frozenset[str] = _ALL_CCD | frozenset({'vgiss'})


@dataclass(frozen=True)
class ModeParam:
    """One parameter of an artifact mode.

    Parameters:
        name: The parameter key inside the mode's scene map.
        kind: The value's type tag, one of ``bool``, ``nonneg_number``,
            ``unit_interval``, ``int``, ``nonneg_int``, ``positive_int``,
            ``enum`` (with ``choices``), or ``int_list`` (with ``length``).
        default: The value used when the key is absent.  ``None`` marks an
            optional parameter with no default shape (the renderer supplies its
            own fallback, e.g. a centred window).
        choices: The permitted values for an ``enum`` parameter.
        length: The required length of an ``int_list`` parameter.
    """

    name: str
    kind: str
    default: Any = None
    choices: tuple[Any, ...] | None = None
    length: int | None = None


# The incidence parameter every mode carries: expected events per frame (count
# modes) or per-frame activation probability (commanded / periodic modes),
# disabled at 0.
_INCIDENCE = ModeParam('incidence', 'nonneg_number', 0.0)


@dataclass(frozen=True)
class ArtifactMode:
    """A registered artifact mode.

    Parameters:
        name: The registry key (also the ``artifacts`` block key).
        stage: The rendering stage that applies it, ``telemetry`` or
            ``detector``.
        implemented: Whether the renderer implements it yet.  An unimplemented
            mode is a reserved registry entry: it fixes the name, stage, and
            availability so the eventual implementation drops in without a
            schema change, and it fails validation with a clear message until
            its rendering code lands.
        params: The mode's parameters (``incidence`` first, then mode-specific).
        availability: The sim instruments the mode is available on (the generic
            block is always accepted, so it is never listed here).
        incidence_semantics: Human-readable meaning of ``incidence`` for the
            mode, for the validator's messages and the developer guide.
        unavailable_reason: Per-instrument bespoke unavailability messages
            (e.g. LORRI hot pixels); other instruments get a generic message.
    """

    name: str
    stage: str
    implemented: bool
    params: tuple[ModeParam, ...]
    availability: frozenset[str]
    incidence_semantics: str
    unavailable_reason: Mapping[str, str] = field(default_factory=dict)

    @property
    def param_map(self) -> dict[str, ModeParam]:
        """The mode's parameters keyed by name."""
        return {param.name: param for param in self.params}


def _telemetry_mode(
    name: str,
    *extra_params: ModeParam,
    availability: frozenset[str],
    incidence_semantics: str,
    unavailable_reason: Mapping[str, str] | None = None,
) -> ArtifactMode:
    """Build an implemented telemetry-stage loss mode with an incidence param."""
    return ArtifactMode(
        name=name,
        stage='telemetry',
        implemented=True,
        params=(_INCIDENCE, *extra_params),
        availability=availability,
        incidence_semantics=incidence_semantics,
        unavailable_reason=unavailable_reason or {},
    )


def _reserved_mode(
    name: str,
    stage: str,
    *,
    availability: frozenset[str],
) -> ArtifactMode:
    """Build a reserved (not-yet-implemented) mode: name, stage, availability.

    The parameter schema is deliberately minimal (incidence only): validation
    rejects a reserved mode as unimplemented before it ever reaches the
    parameters, so the eventual implementation defines the real schema.
    """
    return ArtifactMode(
        name=name,
        stage=stage,
        implemented=False,
        params=(_INCIDENCE,),
        availability=availability,
        incidence_semantics='reserved; defined when the mode is implemented',
    )


# ---------------------------------------------------------------------------
# The registry.
# ---------------------------------------------------------------------------

_IMPLEMENTED_MODES: tuple[ArtifactMode, ...] = (
    _telemetry_mode(
        'missing_lines',
        ModeParam('contiguous_run', 'bool', False),
        availability=frozenset({'coiss_nac', 'coiss_wac', 'vgiss', 'gossi', 'nhlorri'}),
        incidence_semantics='expected number of whole lines lost per frame (Poisson)',
    ),
    _telemetry_mode(
        'partial_lines',
        ModeParam('max_surviving_segments', 'positive_int', 1),
        availability=frozenset({'coiss_nac', 'coiss_wac', 'vgiss'}),
        incidence_semantics='expected number of truncated lines per frame (Poisson)',
    ),
    _telemetry_mode(
        'alternating_lines',
        ModeParam('period', 'enum', 2, choices=(2, 4)),
        ModeParam('phase', 'nonneg_int', 0),
        availability=frozenset({'coiss_nac', 'coiss_wac', 'gossi'}),
        incidence_semantics='per-frame probability the periodic line dropout is active',
    ),
    _telemetry_mode(
        'edited_frame',
        ModeParam('band_width_px', 'positive_int', None),
        ModeParam('half_frame', 'bool', False),
        ModeParam('half', 'enum', 'top', choices=('top', 'bottom')),
        availability=frozenset({'vgiss', 'gossi'}),
        incidence_semantics='per-frame probability the commanded edit is applied',
    ),
    _telemetry_mode(
        'truncated_frame',
        ModeParam('fraction', 'unit_interval', None),
        ModeParam('lines', 'nonneg_int', None),
        ModeParam('from', 'enum', 'bottom', choices=('bottom', 'top')),
        availability=frozenset({'coiss_nac', 'coiss_wac', 'gossi'}),
        incidence_semantics='per-frame probability the frame is truncated',
    ),
    _telemetry_mode(
        'missing_blocks',
        ModeParam('block_lines', 'positive_int', 8),
        ModeParam('start_mid_line', 'bool', False),
        availability=frozenset({'gossi', 'coiss_nac', 'coiss_wac'}),
        incidence_semantics='expected number of compression blocks lost per frame (Poisson)',
    ),
    _telemetry_mode(
        'line_garble',
        availability=frozenset({'vgiss', 'gossi'}),
        incidence_semantics='expected number of garbled lines per frame (Poisson)',
    ),
    _telemetry_mode(
        'pixel_spikes',
        ModeParam('amplitude', 'enum', 'bitflip', choices=('bitflip', 'uniform')),
        availability=frozenset({'vgiss'}),
        incidence_semantics='expected number of spiked pixels per frame (Poisson)',
    ),
    _telemetry_mode(
        'dead_pixels',
        ModeParam('count', 'nonneg_int', None),
        ModeParam('low_dn', 'nonneg_number', 0.0),
        availability=frozenset({'coiss_nac', 'coiss_wac', 'gossi', 'nhlorri'}),
        incidence_semantics='expected number of dead pixels per frame (Poisson) when count absent',
    ),
    _telemetry_mode(
        'dead_columns',
        ModeParam('count', 'nonneg_int', None),
        ModeParam('low_dn', 'nonneg_number', 0.0),
        availability=frozenset({'coiss_nac', 'coiss_wac', 'gossi', 'nhlorri'}),
        incidence_semantics='expected number of dead columns per frame (Poisson) when count absent',
    ),
    _telemetry_mode(
        'embedded_header',
        ModeParam('header_px', 'positive_int', 34),
        availability=frozenset({'nhlorri'}),
        incidence_semantics='per-frame probability the row-0 housekeeping header is written',
    ),
    _telemetry_mode(
        'truth_window',
        ModeParam('size', 'positive_int', 96),
        ModeParam('position', 'int_list', None, length=2),
        availability=frozenset({'gossi'}),
        incidence_semantics='per-frame probability the losslessly-clean carve-out is commanded',
    ),
    _telemetry_mode(
        'cutout_window',
        ModeParam('rect', 'int_list', None, length=4),
        availability=frozenset({'gossi', 'nhlorri'}),
        incidence_semantics='per-frame probability the commanded cut-out window is applied',
    ),
    # hot_pixels is implemented, but on the DETECTOR stage: the registry routes
    # its parameters to the detector hot-pixel population (params.py) rather
    # than to a telemetry applier.  incidence is the fraction of pixels that are
    # hot (a spatial density, not a per-frame event count).
    ArtifactMode(
        name='hot_pixels',
        stage='detector',
        implemented=True,
        params=(
            _INCIDENCE,
            ModeParam('amplitude_e', 'nonneg_number', None),
            ModeParam('column_factor', 'unit_interval', None),
        ),
        availability=frozenset({'coiss_nac', 'coiss_wac', 'gossi'}),
        incidence_semantics='fraction of pixels that are hot (spatial density)',
        unavailable_reason={
            'nhlorri': 'explicitly disabled for LORRI, which has none',
        },
    ),
)

# Reserved detector / electronics modes: named, staged, and availability-scoped
# now so they drop in without a schema change; validation rejects them as
# unimplemented until their rendering code lands (the next sub-delivery).
_RESERVED_MODES: tuple[ArtifactMode, ...] = (
    _reserved_mode('compression_dct', 'telemetry', availability=_ALL_CCD),
    _reserved_mode('reseau_scars', 'telemetry', availability=frozenset({'vgiss'})),
    _reserved_mode('resample_texture', 'detector', availability=frozenset({'vgiss'})),
    _reserved_mode('banding_coherent', 'detector', availability=_ALL_INSTRUMENTS),
    _reserved_mode('bias_structure', 'detector', availability=_ALL_INSTRUMENTS),
    _reserved_mode('dark_ramp', 'detector', availability=_ALL_INSTRUMENTS),
    _reserved_mode(
        'bright_dark_pairs', 'detector', availability=frozenset({'coiss_nac', 'coiss_wac'})
    ),
    _reserved_mode(
        'bloom', 'detector', availability=frozenset({'coiss_nac', 'coiss_wac', 'gossi'})
    ),
    _reserved_mode(
        'quantization_lut', 'detector', availability=frozenset({'coiss_nac', 'coiss_wac'})
    ),
    _reserved_mode(
        'quantization_ls8b', 'detector', availability=frozenset({'coiss_nac', 'coiss_wac'})
    ),
    _reserved_mode('contouring_8bit', 'detector', availability=frozenset({'gossi', 'vgiss'})),
    _reserved_mode('fixed_pattern', 'detector', availability=_ALL_INSTRUMENTS),
    _reserved_mode('radiation_transients', 'detector', availability=_ALL_CCD),
    _reserved_mode('frame_transfer_smear', 'detector', availability=frozenset({'nhlorri'})),
    _reserved_mode('serial_tail', 'detector', availability=frozenset({'nhlorri'})),
    _reserved_mode('beam_bend', 'detector', availability=frozenset({'vgiss'})),
    _reserved_mode('residual_image', 'detector', availability=frozenset({'vgiss'})),
)

ARTIFACT_MODES: dict[str, ArtifactMode] = {
    mode.name: mode for mode in (*_IMPLEMENTED_MODES, *_RESERVED_MODES)
}

# The complete key set the ``artifacts`` block may carry beyond the two switch
# keys.  The validator keys on this exact set.
MODE_KEYS: frozenset[str] = frozenset(ARTIFACT_MODES)

# The fixed order the telemetry stage applies its implemented loss modes in
# (3.3: frame-level commanded shapes, then line losses, then block losses, then
# garble, then per-pixel losses, then the row-0 header last).  ``truth_window``
# is not here: it is a protective carve-out resolved before the loop and passed
# to ``missing_blocks``, not a signal-mutating loss.  ``hot_pixels`` is not here
# either: it is a detector-stage mode.
STRUCTURED_LOSS_ORDER: tuple[str, ...] = (
    'cutout_window',
    'edited_frame',
    'truncated_frame',
    'missing_lines',
    'partial_lines',
    'alternating_lines',
    'missing_blocks',
    'line_garble',
    'dead_columns',
    'pixel_spikes',
    'dead_pixels',
    'embedded_header',
)


def normalize_instrument(instrument: str | None) -> str:
    """Collapse an instrument name to its availability key.

    The calibrated Cassini aliases share the raw detector's availability, and
    the generic aliases (and ``None``) map to ``generic``, which accepts every
    mode.

    Parameters:
        instrument: A sim instrument name, a generic alias, or ``None``.

    Returns:
        The normalized availability key.
    """
    if instrument is None or instrument in ('generic', 'sim'):
        return 'generic'
    aliases = {'coiss_calib_nac': 'coiss_nac', 'coiss_calib_wac': 'coiss_wac'}
    return aliases.get(instrument, instrument)


def mode_available(mode_name: str, instrument: str | None) -> bool:
    """Whether ``mode_name`` is available on ``instrument``.

    Parameters:
        mode_name: A registered mode name.
        instrument: A sim instrument name, a generic alias, or ``None``.

    Returns:
        True if the generic block is selected (which accepts every mode) or the
        instrument is in the mode's availability set.
    """
    norm = normalize_instrument(instrument)
    if norm == 'generic':
        return True
    return norm in ARTIFACT_MODES[mode_name].availability


def mode_unavailable_message(mode_name: str, instrument: str | None) -> str:
    """A validation message explaining why a mode is unavailable on an instrument."""
    norm = normalize_instrument(instrument)
    mode = ARTIFACT_MODES[mode_name]
    bespoke = mode.unavailable_reason.get(norm)
    if bespoke is not None:
        return f'artifact mode {mode_name!r} is {bespoke}'
    available = sorted(mode.availability)
    return (
        f'artifact mode {mode_name!r} is not available for instrument {instrument!r}; '
        f'available on: {available}'
    )


def resolve_mode_config(mode_name: str, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Fill a mode's scene map with its parameter defaults for rendering.

    Parameters:
        mode_name: A registered mode name.
        raw_config: The scene's map for the mode (already validated).

    Returns:
        A fresh dict carrying every parameter, scene value overriding default.
    """
    mode = ARTIFACT_MODES[mode_name]
    resolved: dict[str, Any] = {}
    for param in mode.params:
        resolved[param.name] = raw_config.get(param.name, param.default)
    return resolved
