"""Per-NavStatusReason operator-readable INFO log templates.

The orchestrator emits one INFO line per status_reason summarizing the
final outcome.  Templates here let tests assert the operator-readable
narrative for every reason.
"""

from nav.support.status_reason import NavStatusReason

__all__ = ['STATUS_REASON_INFO_TEMPLATE']


STATUS_REASON_INFO_TEMPLATE: dict[NavStatusReason, list[str]] = {
    NavStatusReason.OK: [
        'Final: status=ok',
    ],
    NavStatusReason.RANK_1_ONLY: [
        'Final: status=ok',
        'Result is rank-1: one axis unobservable',
    ],
    NavStatusReason.CONFLICTED_TECHNIQUES: [
        'Final: status=conflicted',
        'Best-vs-runner-up confidence gap below threshold',
    ],
    NavStatusReason.NO_SIGNAL_IN_IMAGE: [
        'Final: status=failed',
        'Image classifier: blank / dark frame',
    ],
    NavStatusReason.IMAGE_OVEREXPOSED: [
        'Final: status=failed',
        'Image classifier: > 80% pixels at full-well DN',
    ],
    NavStatusReason.MISSING_DATA_DOMINANT: [
        'Final: status=failed',
        'Image classifier: > 30% pixels at missing-data marker',
    ],
    NavStatusReason.IMAGE_CORRUPT: [
        'Final: status=failed',
        'Image file failed to parse / read',
    ],
    NavStatusReason.KERNELS_UNAVAILABLE: [
        'Final: status=failed',
        'SPICE coverage missing for the image ET',
    ],
    NavStatusReason.INSTRUMENT_NOT_CONFIGURED: [
        'Final: status=failed',
        'No config block for this instrument camera',
    ],
    NavStatusReason.NO_FEATURES_EXTRACTED: [
        'Final: status=failed',
        'No extractor produced a feature',
    ],
    NavStatusReason.ALL_FEATURES_GATED: [
        'Final: status=failed',
        'Every feature dropped by the reliability gate',
    ],
    NavStatusReason.NO_FEASIBLE_TECHNIQUES: [
        'Final: status=failed',
        "No technique's is_feasible returned True",
    ],
    NavStatusReason.ALL_TECHNIQUES_SPURIOUS: [
        'Final: status=failed',
        'Every technique returned spurious=True',
    ],
    NavStatusReason.FINAL_CONFIDENCE_BELOW_THRESHOLD: [
        'Final: status=failed',
        'Combined confidence below min_confidence',
    ],
    NavStatusReason.UNOBSERVABLE_OFFSET: [
        'Final: status=failed',
        'Every input covariance shares one null direction',
    ],
}
