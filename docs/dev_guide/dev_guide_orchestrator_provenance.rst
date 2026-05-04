==========================================================
Reproducibility Envelope (Provenance)
==========================================================

Overview
========

:class:`~nav.nav_orchestrator.provenance.Provenance` is the frozen dataclass attached to
every :class:`~nav.nav_orchestrator.nav_result.NavResult` to record the exact runtime
state under which a navigation produced its outputs.  Two navigations with identical
inputs produce byte-identical
:class:`~nav.nav_orchestrator.provenance.Provenance` *except* for
:attr:`~nav.nav_orchestrator.provenance.Provenance.pipeline_run_iso8601`, which is
wall-clock by construction; regression-baseline comparison strips that field before
comparing.

Theory
======

The provenance envelope captures three independent kinds of state:

- **Code state.**  ``rms_nav_version`` and ``rms_nav_git_sha`` together identify the exact
  source code that ran.
- **External-data state.**  ``spice_kernels`` lists every SPICE kernel actually loaded;
  ``static_data_hashes`` sha256-hashes every YAML in
  ``src/nav/config_files`` whose filename matches one of the
  ``_STATIC_DATA_PREFIXES`` (``config_220_`` for the body shape catalogue, ``config_3``
  for ring catalogues, ``config_4`` for per-instrument blocks).
- **Pipeline state.**  ``technique_names`` and ``extractor_names`` enumerate every
  registered technique and NavModel under the current process — so a regression run
  pinned to an old code revision but with a new technique registered records the
  difference in its provenance even when the outputs are otherwise byte-identical.

Restrictions and assumptions
----------------------------

- The static-data hash list is built once per ``navigate`` call by
  :func:`~nav.nav_orchestrator.provenance.collect_provenance_metadata`.  Comments and
  whitespace are included in the hashed bytes, so a YAML edit that only adds a comment
  changes the hash.
- The SPICE-kernel list is read live from ``spiceypy.ktotal`` /
  ``spiceypy.kdata``; the orchestrator does not coerce or sort the list itself
  (the dataclass sorts it for byte-identical output).
- The git SHA is read from ``git rev-parse HEAD`` plus a ``--is-dirty`` check; the
  reported value is ``'dirty'`` when the working tree has uncommitted changes and
  ``None`` when neither git nor a recorded SHA is available.
- The dataclass is frozen; the
  :attr:`~nav.nav_orchestrator.provenance.Provenance.spice_kernel_count` derived field is
  populated in ``__post_init__`` from the kernel list length.

Sources of uncertainty
----------------------

The envelope reports no uncertainty.  Every field is a deterministic readout of runtime
state at navigate time.

Configuration
=============

The envelope carries no YAML configuration of its own.  The list of filename prefixes
counted as static data lives in module-level
``_STATIC_DATA_PREFIXES``; downstream callers that want a different set of YAML files
hashed must extend the list at module level.

Implementation
==============

Source file: ``src/nav/nav_orchestrator/provenance.py`` —
:class:`~nav.nav_orchestrator.provenance.Provenance`,
:class:`~nav.nav_orchestrator.provenance.ProvenanceMetadata`, and
:func:`~nav.nav_orchestrator.provenance.collect_provenance_metadata`.

Public surface (autodocumented at :doc:`/api_reference/api_nav_orchestrator`):

- :class:`~nav.nav_orchestrator.provenance.Provenance` — frozen dataclass.  Public fields:

  - :attr:`~nav.nav_orchestrator.provenance.Provenance.rms_nav_version` — version string
    (e.g. ``'0.5.2'``).
  - :attr:`~nav.nav_orchestrator.provenance.Provenance.image_et` — observation midtime ET
    (TDB seconds past J2000).
  - :attr:`~nav.nav_orchestrator.provenance.Provenance.pipeline_run_iso8601` — UTC timestamp
    when the run began; excluded from regression-baseline comparison.
  - :attr:`~nav.nav_orchestrator.provenance.Provenance.rms_nav_git_sha` — short git SHA,
    ``'dirty'``, or ``None``.
  - :attr:`~nav.nav_orchestrator.provenance.Provenance.spice_kernels` — sorted tuple of
    SPICE kernel filenames.
  - :attr:`~nav.nav_orchestrator.provenance.Provenance.static_data_hashes` — read-only
    mapping of YAML filename to sha256 hex digest.
  - :attr:`~nav.nav_orchestrator.provenance.Provenance.technique_names` — sorted tuple of
    registered technique class names.
  - :attr:`~nav.nav_orchestrator.provenance.Provenance.extractor_names` — sorted tuple of
    registered extractor class names.
  - :attr:`~nav.nav_orchestrator.provenance.Provenance.spice_kernel_count` — derived;
    populated from ``len(spice_kernels)`` in ``__post_init__``.

- :class:`~nav.nav_orchestrator.provenance.ProvenanceMetadata` — internal dataclass
  returned by
  :func:`~nav.nav_orchestrator.provenance.collect_provenance_metadata` carrying the
  freshly-read git SHA, kernel list, and static-data hash dict.

- :func:`~nav.nav_orchestrator.provenance.collect_provenance_metadata` — runs the live
  readouts.  Called once per
  :meth:`~nav.nav_orchestrator.orchestrator.NavOrchestrator.navigate`.

The dataclass enforces invariants in ``__post_init__``: every collection input is coerced
to its read-only / sorted form so two
:class:`~nav.nav_orchestrator.provenance.Provenance` instances with the same inputs are
byte-identical for hash / serialisation.

Examples
========

**Two navigations of the same image with the same SPICE kernels.**  An operator runs a
batch over a Cassini ISS image at two different wall-clock times.  Both runs produce
:class:`~nav.nav_orchestrator.provenance.Provenance` instances that differ only in
:attr:`~nav.nav_orchestrator.provenance.Provenance.pipeline_run_iso8601`; every other
field is byte-identical.  A regression-baseline comparator strips that field and confirms
the two outputs match.

**Code change that surfaces in provenance.**  An operator pulls a new commit that touches
:mod:`nav.nav_technique.dt_fitting` and reruns navigation.  The new
:class:`~nav.nav_orchestrator.provenance.Provenance` carries a different
:attr:`~nav.nav_orchestrator.provenance.Provenance.rms_nav_git_sha`; the rest of the
envelope is unchanged unless the per-technique result changed.  The reviewer can correlate
output diffs with the SHA delta directly.

**Static-data change.**  An operator edits ``config_220_body_shape.yaml`` to refine
Mimas's ellipsoid residual.  The next
:class:`~nav.nav_orchestrator.provenance.Provenance` carries a different
:attr:`~nav.nav_orchestrator.provenance.Provenance.static_data_hashes` entry for that
file; downstream regression baselines pinned to the old hash flag the difference and
require re-baselining.
