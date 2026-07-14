"""NavStatusReason — typed enumeration of every navigation outcome reason.

A single ``NavStatusReason`` value is attached to every ``NavResult`` produced by
the orchestrator.  The value tells callers, in one slot, exactly *why* the
result has its status — success modes, image-quality refusals, kernel
problems, ensemble disagreement, and degenerate-geometry refusals.

Located under ``spindoctor.support`` (not ``spindoctor.nav_orchestrator``) because some values
describe failure modes that are not strictly navigation outcomes — image-load
errors, missing kernels, instrument-not-configured — and are produced by code
outside the orchestrator (image-quality classifier, kernel-loading shim,
dataset enumeration).
"""

from enum import StrEnum

__all__ = ['NavStatusReason']


class NavStatusReason(StrEnum):
    """Discrete outcome reasons set on every NavResult.

    The value tells downstream consumers exactly why the navigation has its
    final status.  Each value maps to a specific outcome:

    - ``OK``: normal success.
    - ``RANK_1_ONLY``: success with only one observable axis (e.g. flat-ring
      scene with no orthogonal feature); ``sigma_along_unobservable_px`` is
      set on the result.
    - ``CONFLICTED_TECHNIQUES``: multiple agreement groups exist after
      grouping and the best-vs-runner-up summed-confidence gap is below
      ``agreement_gap``; offset reported with reduced confidence.
    - ``NO_SIGNAL_IN_IMAGE``: image classifier flagged a blank or dark frame.
    - ``IMAGE_OVEREXPOSED``: image classifier saw most pixels at full-well DN.
    - ``MISSING_DATA_DOMINANT``: image classifier saw too many missing-data
      pixels.
    - ``IMAGE_CORRUPT``: image file failed to parse or read.
    - ``KERNELS_UNAVAILABLE``: SPICE coverage missing for the image ET.
    - ``INSTRUMENT_NOT_CONFIGURED``: no per-instrument YAML block for this
      camera.
    - ``TITAN_UNSUPPORTED``: the only navigable content in the frame is Titan,
      whose opaque haze hides the surface, so no shape-based or haze-limb
      navigation is available; the pipeline records the refusal instead of a
      silent empty failure.
    - ``NO_FEATURES_EXTRACTED``: every extractor returned an empty list.
    - ``ALL_FEATURES_GATED``: features extracted but every one fell below the
      reliability gate.
    - ``NO_FEASIBLE_TECHNIQUES``: features pass the gate but no technique's
      ``is_feasible`` returned true.
    - ``ALL_TECHNIQUES_SPURIOUS``: every technique returned ``spurious=True``.
    - ``FINAL_CONFIDENCE_BELOW_THRESHOLD``: ensemble combined confidence sat
      below ``min_confidence`` (or below every tier's ``min_confidence``).
    - ``FINAL_SIGMA_ABOVE_THRESHOLD``: combined confidence cleared the lowest
      tier's ``min_confidence`` but the offset sigma exceeded every tier's
      ``max_sigma_px`` (confident but too imprecise to earn any tier).
    - ``UNOBSERVABLE_OFFSET``: every input covariance shares one null
      direction; the precision-weighted combine cannot proceed.
    - ``CONTRACT_VIOLATION``: an internal navigation invariant was violated
      (``NavContractError``); a programming error upstream, not bad image
      data.  The full traceback is in the error log.
    """

    OK = 'ok'
    RANK_1_ONLY = 'rank_1_only'
    CONFLICTED_TECHNIQUES = 'conflicted_techniques'
    NO_SIGNAL_IN_IMAGE = 'no_signal_in_image'
    IMAGE_OVEREXPOSED = 'image_overexposed'
    MISSING_DATA_DOMINANT = 'missing_data_dominant'
    IMAGE_CORRUPT = 'image_corrupt'
    KERNELS_UNAVAILABLE = 'kernels_unavailable'
    INSTRUMENT_NOT_CONFIGURED = 'instrument_not_configured'
    TITAN_UNSUPPORTED = 'titan_unsupported'
    NO_FEATURES_EXTRACTED = 'no_features_extracted'
    ALL_FEATURES_GATED = 'all_features_gated'
    NO_FEASIBLE_TECHNIQUES = 'no_feasible_techniques'
    ALL_TECHNIQUES_SPURIOUS = 'all_techniques_spurious'
    FINAL_CONFIDENCE_BELOW_THRESHOLD = 'final_confidence_below_threshold'
    FINAL_SIGMA_ABOVE_THRESHOLD = 'final_sigma_above_threshold'
    UNOBSERVABLE_OFFSET = 'unobservable_offset'
    CONTRACT_VIOLATION = 'contract_violation'
