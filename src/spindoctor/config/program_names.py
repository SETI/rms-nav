"""Canonical program identities for the SpinDoctor command-line programs.

A program identity names one program for two purposes: it is the directory the
program's main log is written under, and it is the key a ``logging.programs``
configuration block is looked up by.  Both come from the same constant so the
two can never drift apart.

Several dispatch modules share an identity.  The three mosaic entry points are
one program with mode variants, and each cloud-task driver shares the identity
of the interactive driver it mirrors, so that one configuration block governs
both and their per-image logs land in the same backend subtree.

Programs that carry no logger at all -- the statistics report and the graphical
programs, which write to the terminal directly -- have no identity here,
because they have nothing to name.
"""

SD_BACKPLANES = 'sd_backplanes'
SD_CONSOLIDATE_METADATA = 'sd_consolidate_metadata'
SD_CREATE_BUNDLE = 'sd_create_bundle'
SD_CREATE_CK = 'sd_create_ck'
SD_MOSAIC = 'sd_mosaic'
SD_OFFSET = 'sd_offset'
SD_RESULTS_INDEX = 'sd_results_index'

PROGRAM_NAMES = frozenset(
    {
        SD_BACKPLANES,
        SD_CONSOLIDATE_METADATA,
        SD_CREATE_BUNDLE,
        SD_CREATE_CK,
        SD_MOSAIC,
        SD_OFFSET,
        SD_RESULTS_INDEX,
    }
)
"""Every valid program identity, used to validate ``logging.programs`` keys."""
